$env:PYTHONUTF8 = "1"
# Keep packages in the user folder (AppData\Roaming\Python) from shadowing the environment.
$env:PYTHONNOUSERSITE = "1"
conda run -n aabundle --no-capture-output python -X utf8 @args
exit $LASTEXITCODE
