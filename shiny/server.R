shinyServer(function(input, output, session) {
  session$onSessionEnded(
    # nothing to do for now..
    function() {}
  )
  
  for (f in list.files('server', pattern = '.R', full.names = T)) {
    source(f, local = TRUE)
  }
  
  # to trigger preparing the initial job table
  trigger_update_job_table(rnorm(1))
})
