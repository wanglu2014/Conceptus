
cat("Testing R environment...\n")
cat("Library path:", .libPaths(), "\n")

cat("Attempting to load SpiecEasi...\n")
tryCatch({
    library(SpiecEasi)
    cat("SpiecEasi loaded successfully!\n")
    
    # Test internal function availability (check DLL loading)
    cat("Checking spiec.easi function...\n")
    if (exists("spiec.easi")) {
        cat("spiec.easi function found.\n")
    } else {
        cat("spiec.easi function NOT found.\n")
    }

}, error = function(e) {
    cat("ERROR loading SpiecEasi:\n")
    print(e)
})
