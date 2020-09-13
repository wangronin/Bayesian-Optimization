job_list <- reactiveValues(list = list())
tell_done <- reactiveValues(value = 0)

search_space <- reactive({
  req(input$upload.add_zip$datapath)
  json_file <- input$upload.add_zip$datapath
  json_data <- fromJSON(file = json_file)
})

job_table <- reactive({
  if (length(job_list$list) == 0) return(data.table())
  .json <- list(
    check_job = as.vector(job_list$list)
  )
  r <- POST("http://207.246.97.250:7200", body = .json, encode = 'json')
  as.data.frame(do.call(rbind, content(r)))
})

observeEvent(input$upload.create_job, {
  .cs <- search_space()
  
  if (length(.cs) != 0) {
    r <- POST("http://207.246.97.250:7200", body = .cs, encode = 'json')
    if (status_code(r) == 200) {
      job_id <- content(r)$job_id
      job_list$list <- c(job_list$list, job_id)
      
      print_html(
        paste0(
          '<p style="color:red;">成功创建',
          job_id,
          '任务!</p>'
        )
      )
    } else {
      print_html('<p style="color:red;">任务创建失败!</p>')
    }
  }
})

observeEvent(input$upload.delete_job, {
  req(input$job_list_delete)
  job_id <- input$job_list_delete
  r <- GET(
    "http://207.246.97.250:7200", 
    query = list(finalize = 'null', job_id = job_id)
  )
  
  if (status_code(r) == 200) {
    job_list$list <- setdiff(job_list$list, list(job_id))
    print_html2(
      paste0(
        '<p style="color:red;">成功删除',
        job_id,
        '任务!</p>'
      )
    )
  } else {
    print_html2('<p style="color:red;">删除任务失败!</p>')
  }
})

ask_table <- eventReactive(input$ask_box.ask, {
  job_id <- input$ask_box.job_id
  r <- GET(
    "http://207.246.97.250:7200", 
    query = list(ask = 'null', job_id = job_id)
  )
  
  if (status_code(r) == 200) {
    df <- as.data.frame(do.call(rbind, content(r)['X'][[1]]))
    print_html(
      paste0(
        '<p style="color:red;">成功删除',
        job_id,
        '任务!</p>'
      )
    )
  } else {
    df <- data.table()
    print_html('<p style="color:red;">删除任务失败!</p>')
  }
  df
})

observeEvent(input$tell_box.tell, {
  job_id <- input$tell_box.job_id

  req(input$tell_box.add_json$datapath)
  json_data <- jsonlite::read_json(input$tell_box.add_json$datapath)
  json_data[['job_id']] <- job_id
  
  r <- POST(
    "http://207.246.97.250:7200",
    body = json_data, encode = 'json'
  )

  if (status_code(r) == 200) {
    print_html(
      paste(
        '<p style="color:red;">成功添加目标值',
        paste(json_data$y, collapse = ', '),
        '至任务',
        job_id,
        '</p>'
      )
    )
    tell_done$value <- rnorm(1)
  } else {
    print_html('<p style="color:red;">成功添加目标值失败!</p>')
  }
})

observe({
  req(job_list$list)
  .list <- job_list$list
  updateSelectInput(session, 'job_list_delete', choices = .list, selected = .list[1])
  updateSelectInput(session, 'ask_box.job_id', choices = .list, selected = .list[1])
  updateSelectInput(session, 'tell_box.job_id', choices = .list, selected = .list[1])
  updateSelectInput(session, 'feature_importance_box.job_id', choices = .list, selected = .list[1])
})

output$data_info <- renderDataTable({
  job_table()
}, options = list(
  pageLength = 10, scrollX = F, autoWidth = TRUE,
  columnDefs = list(list(width = '20px', targets = c(0, 1))))
)

output$ask_data <- renderDataTable({
  ask_table()
}, options = list(
  pageLength = 10, scrollX = F, autoWidth = TRUE,
  columnDefs = list(list(width = '20px', targets = c(0, 1))))
)

output$ask_download <- downloadHandler(
  filename = function() {
    'solutions.json'
  },
  content = function(file) {
    df <- ask_table()
    jsonlite::write_json(list(X = df), file, auto_unbox = T)
  }
)
