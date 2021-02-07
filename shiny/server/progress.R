output$progress_box.line_chart <- renderPlotly({
  data <- progress_data()
  p <- plotly_default('任务性能', '迭代次数', '目标值')
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
  req(tell_lock$value == 0)
  isolate({
    job_list <- rownames(job_table())
    .json <- list(
      get_history = as.vector(job_list)
    )
  })
  r <- POST(address, body = .json, encode = 'json')
  content(r)
})