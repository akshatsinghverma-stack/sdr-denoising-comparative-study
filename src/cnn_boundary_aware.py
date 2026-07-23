"""
Boundary-Aware CNN Denoising Autoencoder
==========================================
Diagnostic follow-up to `src/cnn_autoencoder.py`.

Case Study 1 (report.md Section 3.5) found that the plain-MSE CNN posts large
apparent SNR gains yet *loses* to doing-nothing on BER, at every SNR level
tested. The report's explanation: plain MSE is a smooth, symmetric,
threshold-blind loss with no explicit penalty for crossing the decision
boundary (the sign of each I/Q component for BPSK/QPSK) -- so during training
it can trade a handful of sign flips near zero for a larger aggregate
reduction in squared error elsewhere in the batch.

This module makes that a *tested* claim rather than an observation: it adds a
boundary-aware loss term -- a soft-hinge margin penalty on top of MSE -- that
explicitly penalizes outputs landing on the wrong side of (or too close to)
the true decision boundary, and provides a training/build API that mirrors
`cnn_autoencoder.py`'s pattern so the two can be trained and compared
side-by-side on identical data.

Architecture is identical to `build_autoencoder` in cnn_autoencoder.py (same
layer sizes/shapes) -- only the loss function differs. This isolates the loss
function as the sole independent variable in the comparison.

Nothing in `cnn_autoencoder.py` is modified; this file only imports its
(unmodified) helper functions for data prep / windowing / reconstruction.
"""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # suppress TF info/warning spam

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Re-use the (unmodified) data-prep / windowing / reconstruction helpers so
# both loss variants operate on exactly the same data pipeline.
from src.cnn_autoencoder import (  # noqa: F401
    _iq_to_real, _real_to_iq, make_windows, reconstruct_from_windows,
    prepare_dataset, denoise_signal,
)


# ---------------------------------------------------------------------------
# Boundary-aware loss functions
# ---------------------------------------------------------------------------

def make_boundary_hinge_loss(margin: float = 0.3, hinge_weight: float = 1.0):
    """Build a Keras loss: MSE + a soft-hinge margin penalty.

    For BPSK/QPSK, the true decision boundary for each I/Q component is
    exactly zero -- the sign of the clean target `y_true` tells you which
    side of the boundary the symbol is *supposed* to land on. The hinge term

        hinge = mean( relu(margin - sign(y_true) * y_pred) )

    is zero whenever the prediction is correctly signed *and* at least
    `margin` away from the boundary, and grows linearly as the prediction
    approaches, crosses, or lands on the wrong side of zero. This is the
    classic (soft-margin SVM style) hinge penalty, applied per I/Q component,
    per sample, per window.

    Channels where the true target is exactly 0 (e.g. BPSK's imaginary
    component, which carries no information and has no decision boundary)
    are masked out of the hinge term via `sign(y_true) == 0`, so the penalty
    only ever fires on components that actually encode a bit.

    Parameters
    ----------
    margin : float
        Minimum acceptable distance from the decision boundary (in the same
        units as the clean target amplitude, e.g. ~1.0 for BPSK, ~0.707 for
        each QPSK component). Higher = more aggressive push away from zero.
    hinge_weight : float
        Weight of the hinge term relative to MSE (which is O(noise power)).
    """

    def _loss(y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))

        sign_true = tf.sign(y_true)
        active_mask = tf.cast(tf.not_equal(sign_true, 0.0), y_pred.dtype)

        margin_violation = tf.nn.relu(margin - sign_true * y_pred) * active_mask
        # Mean over only the "active" (boundary-relevant) entries so that
        # BPSK's always-zero Q channel doesn't dilute -- or, worse, silently
        # zero out via a constant relu(margin) term -- the hinge signal.
        n_active = tf.reduce_sum(active_mask)
        hinge = tf.reduce_sum(margin_violation) / tf.maximum(n_active, 1.0)

        return mse + hinge_weight * hinge

    _loss.__name__ = f"boundary_hinge_loss_m{margin}_w{hinge_weight}"
    return _loss


def make_boundary_bce_loss(bce_weight: float = 1.0, soft_scale: float = 5.0):
    """Build a Keras loss: MSE + a soft binary-cross-entropy demod proxy.

    A differentiable proxy for "did we get the bit right": squash the raw
    output through `tanh(soft_scale * y_pred)` to get a soft decision in
    (-1, 1), rescale to (0, 1), and take BCE against the TRUE bit label
    implied by `sign(y_true)` (available here because training is fully
    supervised against known clean symbols). Like the hinge variant, this is
    masked so channels with an all-zero target (no decision boundary, e.g.
    BPSK's Q channel) don't contribute spurious gradient.

    Parameters
    ----------
    bce_weight : float
        Weight of the BCE term relative to MSE.
    soft_scale : float
        Steepness of the soft-decision sigmoid/tanh -- higher makes the proxy
        closer to a true hard decision (more boundary-sensitive gradient near
        zero), at the cost of vanishing gradients further from the boundary.
    """

    def _loss(y_true, y_pred):
        mse = tf.reduce_mean(tf.square(y_true - y_pred))

        sign_true = tf.sign(y_true)
        active_mask = tf.cast(tf.not_equal(sign_true, 0.0), y_pred.dtype)
        true_label = tf.cast((sign_true + 1.0) / 2.0, y_pred.dtype)  # {0, 1}

        soft_decision = (tf.tanh(soft_scale * y_pred) + 1.0) / 2.0
        soft_decision = tf.clip_by_value(soft_decision, 1e-7, 1.0 - 1e-7)

        bce_elem = -(true_label * tf.math.log(soft_decision) +
                     (1.0 - true_label) * tf.math.log(1.0 - soft_decision))
        bce_elem = bce_elem * active_mask
        n_active = tf.reduce_sum(active_mask)
        bce = tf.reduce_sum(bce_elem) / tf.maximum(n_active, 1.0)

        return mse + bce_weight * bce

    _loss.__name__ = f"boundary_bce_loss_w{bce_weight}_s{soft_scale}"
    return _loss


# ---------------------------------------------------------------------------
# Model definition (architecture identical to build_autoencoder)
# ---------------------------------------------------------------------------

def build_boundary_aware_autoencoder(window_len: int = 128, num_channels: int = 2,
                                     loss_kind: str = "hinge",
                                     margin: float = 0.3, hinge_weight: float = 1.0,
                                     bce_weight: float = 1.0, soft_scale: float = 5.0,
                                     learning_rate: float = 1e-3):
    """Build the boundary-aware CNN denoising autoencoder.

    Identical architecture to `cnn_autoencoder.build_autoencoder` (same
    layer sizes, ~15k parameters) -- only the compiled loss differs, so the
    loss function is the sole independent variable versus the plain-MSE
    baseline.

    Parameters
    ----------
    loss_kind : {"hinge", "bce"}
        Which boundary-aware loss term to add on top of MSE.
    """
    inp = keras.Input(shape=(window_len, num_channels), name="noisy_input")

    x = layers.Conv1D(32, 7, activation="relu", padding="same", name="enc1")(inp)
    x = layers.Conv1D(16, 5, activation="relu", padding="same", name="enc2")(x)
    x = layers.Conv1D(8,  3, activation="relu", padding="same", name="enc3")(x)

    x = layers.Conv1D(16, 3, activation="relu", padding="same", name="dec1")(x)
    x = layers.Conv1D(32, 5, activation="relu", padding="same", name="dec2")(x)
    x = layers.Conv1D(num_channels, 7, activation="linear", padding="same", name="dec_out")(x)

    model = keras.Model(inp, x, name="cnn_denoiser_boundary_aware")

    if loss_kind == "hinge":
        loss_fn = make_boundary_hinge_loss(margin=margin, hinge_weight=hinge_weight)
    elif loss_kind == "bce":
        loss_fn = make_boundary_bce_loss(bce_weight=bce_weight, soft_scale=soft_scale)
    else:
        raise ValueError(f"Unknown loss_kind: {loss_kind!r} (expected 'hinge' or 'bce')")

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss=loss_fn)
    return model


def train_boundary_aware_autoencoder(X_train, Y_train, X_val, Y_val,
                                     window_len: int = 128,
                                     epochs: int = 50, batch_size: int = 64,
                                     loss_kind: str = "hinge",
                                     margin: float = 0.3, hinge_weight: float = 1.0,
                                     bce_weight: float = 1.0, soft_scale: float = 5.0,
                                     verbose: int = 1):
    """Build, train, and return the boundary-aware autoencoder + history.

    Mirrors `cnn_autoencoder.train_autoencoder`'s signature/behaviour
    (same EarlyStopping patience/monitor) so training procedure is not a
    confound in the plain-MSE vs. boundary-aware comparison.
    """
    model = build_boundary_aware_autoencoder(
        window_len=window_len, loss_kind=loss_kind,
        margin=margin, hinge_weight=hinge_weight,
        bce_weight=bce_weight, soft_scale=soft_scale,
    )

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                      restore_best_weights=True),
    ]

    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose,
    )
    return model, history
