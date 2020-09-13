repo_dir <- ''         # repository directory
repo_data <- NULL      # repository data
has_rendered_ERT_per_fct <- FALSE

# Formatter for function values. Somewhat overkill with as.numeric, but this prevents unneeded precision
format_FV <- function(v) as.numeric(format(v, digits = getOption("IOHanalyzer.precision", 2), 
                                nsmall = getOption("IOHanalyzer.precision", 2)))
format_RT <- function(v) as.integer(v)

# directory where data are extracted from the zip file
exdir <- file.path(tempdir(), 'data')

rand_strings <- function(n = 10) {
  a <- do.call(paste0, replicate(5, sample(LETTERS, n, TRUE), FALSE))
  paste0(a, sprintf("%04d", sample(9999, n, TRUE)), sample(LETTERS, n, TRUE))
}

setTextInput <- function(session, id, name, alternative) {
  v <- REG[[id]]
  if (name %in% names(v)) 
    updateTextInput(session, id, value = v[[name]])
  else
    updateTextInput(session, id, value = alternative)
}

# register previous text inputs, which is used to restore those values
REG <- lapply(widget_id, function(x) list())

# TODO: maybe give this function a better name
# get the current 'id' of the selected data: funcID + DIM
get_data_id <- function(dsList) {
  if (is.null(dsList) | length(dsList) == 0)
    return(NULL)

  paste(get_funcId(dsList), get_dim(dsList), sep = '-')
}

# Define server logic required to draw a histogram
shinyServer(function(input, output, session) {
  session$onSessionEnded(function() {
    # close_connection()
    unlink(exdir, recursive = T)
  })
  
  for (f in list.files('server', pattern = '.R', full.names = T)) {
    source(f, local = TRUE)
  }
})
