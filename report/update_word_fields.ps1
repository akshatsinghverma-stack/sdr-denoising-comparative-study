# update_word_fields.ps1 -- bakes real, fully STATIC content into the Word
# report's TOC / List of Figures / List of Tables / page numbers.
#
# python-docx can only ever write the field CODE (e.g. `TOC \h \z \c "Figure"`),
# never the computed RESULT (the actual heading/caption text and page numbers) --
# that computation is Word's own layout engine, which python-docx never runs.
#
# Two problems, addressed in two steps:
#   1. Without ANY update, the field's cached result is empty -- mobile viewers,
#      PDF export paths, and some print drivers only render whatever's already
#      cached and never execute field-code logic, so those pages render blank.
#   2. Baking in the computed result but leaving it as a LIVE field (with
#      updateFields=true telling every opener to recompute it) trades that
#      problem for a different one: weaker mobile field-code engines can choke
#      trying to recompute a TOC on open, which is what caused this report's
#      TOC specifically to still glitch after step 1 alone.
#
# So this script (a) computes real content via Fields.Update()/TOC.Update(),
# then (b) unlinks every field in the document (Fields.Unlink(), equivalent to
# Ctrl+A, Ctrl+Shift+F9) to convert the whole thing to permanent, static text
# and hyperlinks -- no live field codes anywhere, so there is nothing left for
# any renderer to recompute, correctly or not. Internal navigation (clicking a
# TOC entry to jump to that heading) is preserved, since that's a plain
# hyperlink, not a field; only the auto-*updating* behavior is removed, which
# is the right tradeoff for a final, distributed report (the alternative is a
# TOC that silently goes stale or breaks depending on which app opens it).
#
# Requires: Microsoft Word installed locally (uses COM automation). Silently
# does nothing useful if Word isn't installed -- run manually in Word instead
# (Ctrl+A, F9, then Ctrl+A, Ctrl+Shift+F9, then save) if this script isn't
# usable in your environment.

param(
    [Parameter(Mandatory=$true)][string]$Path
)

$resolvedPath = (Resolve-Path $Path).Path
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$doc = $null
try {
    $doc = $word.Documents.Open($resolvedPath)
    $doc.Fields.Update() | Out-Null
    for ($i = 1; $i -le $doc.TablesOfContents.Count; $i++) {
        $doc.TablesOfContents.Item($i).Update()
    }
    $doc.Repaginate() | Out-Null
    $doc.Fields.Update() | Out-Null
    # Convert every field (TOC, List of Figures, List of Tables, page numbers,
    # SEQ-numbered captions) to permanent static text now that the computed
    # values are correct -- removes all live field-code complexity, which is
    # what a weak mobile OOXML parser was choking on even with real cached
    # content already present.
    $doc.Fields.Unlink() | Out-Null
    $doc.Save()
    Write-Output "OK: updated and unlinked (made static) all fields in $resolvedPath"
} finally {
    # Explicit COM release + quit: without this, PowerShell's GC can leave the
    # WINWORD.EXE process resident (invisible, no window) long after this
    # script exits, silently accumulating across repeated runs -- found and
    # cleaned up manually once already, so release deterministically instead
    # of relying on .NET finalizers to eventually get to it.
    if ($doc) {
        $doc.Close([ref]$false) | Out-Null
        [System.Runtime.InteropServices.Marshal]::ReleaseComObject($doc) | Out-Null
    }
    if ($word) {
        $word.Quit()
        [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
    }
    [System.GC]::Collect()
    [System.GC]::WaitForPendingFinalizers()
}
