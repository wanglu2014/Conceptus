
# Set PATH to include Rtools properly (Windows only)
if (.Platform$OS.type == "windows") {
  rtools_paths <- c(
    "Y:/_Software/R/rtools45/usr/bin",
    "Y:/_Software/R/rtools45/x86_64-w64-mingw32.static.posix/bin"
  )
  
  new_path <- paste(c(rtools_paths, Sys.getenv("PATH")), collapse = ";")
  Sys.setenv(PATH = new_path)
  
  # Verify tools are found
  cat("Checking build tools (Windows)...\n")
  cat("Make path:", Sys.which("make"), "\n")
  cat("G++ path:", Sys.which("g++"), "\n")
} else {
  cat("Running on non-Windows (Linux?). Skipping Rtools setup.\n")
}

# Install SpiecEasi
if (!require("devtools", quietly = TRUE)) install.packages("devtools", repos = "https://cloud.r-project.org/")
devtools::install_github("zdk123/SpiecEasi", quiet = FALSE, force = TRUE, upgrade = "never")
