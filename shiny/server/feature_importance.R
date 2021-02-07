output$feature_importance_box.bar_chart <- renderPlotly({
  data <- imp_data()
  p <- plotly_default('', '参数', '参数重要性')
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
  req(tell_lock$value)
  req(tell_lock$value == 0)
  job_id <- input$feature_importance_box.job_id
  .json <- list(
    get_feature_importance = as.list(job_id)
  )
  r <- POST(address, body = .json, encode = 'json')
  content(r)
})