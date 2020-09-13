output$progress_box.line_chart <- renderPlotly({
  data <- progress_data()
  p <- IOH_plot_ly_default('任务性能', '迭代次数', '目标值')
  i <- 1
  
  for (job_id in names(data)) {
    y <- as.vector(data[[job_id]][['hist_f']])
    if (length(y) == 0) next()
    x <- seq(length(y))
    
    p <- add_trace(
      p, x = x, y = y, color = .colors[i %% 5], name = job_id,
      type = 'scatter', mode = 'lines+markers'
    ) 
    i <- i + 1
  }
  p
})

progress_data <- reactive({
  tell_done$value
  isolate({
    if (length(job_list$list) == 0) return(list())
    .json <- list(
      get_history = as.vector(job_list$list)
    )
  })
  
  r <- POST("http://127.0.0.1:7200", body = .json, encode = 'json')
  content(r)
})