# Get the directory of the currently running script. We need it, because
# if the auto generated R file is sourced from a different directory (and assuming the shared
# library is next to the autogen'd R file) the R interpreter won't find it.
# So we need to fix the path based on the location of the autogen'd script.
scriptDir <- dirname(sys.frame(1)$ofile)
# This is a bit of a hack, see:
# https://stackoverflow.com/a/16046056https://stackoverflow.com/a/16046056
# Otherwise we could depend on the `here` library...
# Construct the path to another script in the same directory (or a subdirectory)
libPath <- file.path(scriptDir, "libexample.so")
dyn.load(libPath)
des <- function(dimension, maxGenerations) {
      return(.Call("des", dimension, maxGenerations))

}

