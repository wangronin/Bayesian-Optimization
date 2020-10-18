shinyServer(function(input, output, session) {
  session$onSessionEnded(
    # nothing to do for now..
    function() {}
  )
  
  for (f in list.files('server', pattern = '.R', full.names = T)) {
    source(f, local = TRUE)
  }
})
