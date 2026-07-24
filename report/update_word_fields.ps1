# update_word_fields.ps1 -- bakes real cached content into the Word report's
# TOC / List of Figures / List of Tables field codes.
#
# python-docx can only ever write the field CODE (e.g. `TOC \h \z \c "Figure"`),
# never the computed RESULT (the actual heading/caption text and page numbers) --
# that computation is Word's own layout engine, which python-docx never runs.
# Desktop Word recomputes these on open (this report sets updateFields=true in
# settings.xml for exactly that reason), but many mobile viewers, PDF export
# paths, and some print drivers only render whatever is already cached in the
# file and never execute field-code logic -- so without this step, those pages
# render blank or throw errors downstream (this is what caused mobile/printer
# failures on a version of this report that skipped it).
#
# This script opens the .docx in a real (invisible) Word instance, forces every
# field to recompute, repaginates, and saves -- so the cached result is real
# content, not empty, for every renderer, not just desktop Word.
#
# Requires: Microsoft Word installed locally (uses COM automation). Silently
# does nothing useful if Word isn't installed -- run manually in Word instead
# (Ctrl+A, F9, then save) if this script isn't usable in your environment.

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
    $doc.Save()
    Write-Output "OK: updated TOC/List of Figures/List of Tables fields in $resolvedPath"
} finally {
    if ($doc) { $doc.Close([ref]$false) }
    $word.Quit()
}
