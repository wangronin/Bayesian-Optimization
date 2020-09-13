output$feature_importance_box.bar_chart <- renderPlotly({
  data <- imp_data()
  p <- IOH_plot_ly_default('', '参数', '参数重要性')
  if (is.null(data)) return(p)
  
  i <- 1
  for (job_id in names(data)) {
    x <- unlist(data[[job_id]][['feature']])
    y <- unlist(data[[job_id]][['imp']])
    if (length(x) == 0) next()
    
    p <- add_trace(
      p, x = x, y = y, color = .colors[i %% 5], name = job_id,
      type = 'bar'
    ) 
    i <- i + 1
  }
  p
})

imp_data <- reactive({
  if (tell_done$value == 0) return(NULL)
  job_id <- isolate(input$feature_importance_box.job_id)
  .json <- list(
    get_feature_importance = as.list(job_id)
  )
  r <- POST("http://127.0.0.1:7200", body = .json, encode = 'json')
  content(r)
})