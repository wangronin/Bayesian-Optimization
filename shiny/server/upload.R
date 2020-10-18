job_list <- reactiveValues(list = list())

search_space <- reactive({
  req(input$upload.add_zip$datapath)
  json_data <- jsonlite::read_json(input$upload.add_zip$datapath)
})

job_table <- reactive({
  if (length(job_list$list) == 0) return(data.table())
  .json <- list(
    check_job = as.vector(job_list$list)
  )
  r <- POST(address, body = .json, encode = 'json')
  as.data.frame(do.call(rbind, content(r)))
})

observeEvent(input$upload.create_job, {
  .cs <- search_space()
  
  if (length(.cs) != 0) {
    r <- POST(address, body = .cs, encode = 'json')
    if (status_code(r) == 200) {
      job_id <- content(r)$job_id
      job_list$list <- c(job_list$list, job_id)
      
      print_html(
        paste0(
          '成功创建',
          job_id,
          '任务!'
        )
      )
    } else if (status_code(r) == 500) {
      print_html(
        paste0(
          '<p style="color:red;">任务创建失败:',
          content(r)$error,
          '</p>'
        )
      )
    }
  }
})

observeEvent(input$upload.delete_job, {
  req(input$job_list_delete)
  job_id <- input$job_list_delete
  r <- GET(
    address, 
    query = list(finalize = 'null', job_id = job_id)
  )
  
  if (status_code(r) == 200) {
    job_list$list <- setdiff(job_list$list, list(job_id))
    print_html(
      paste0(
        '<p style="color:red;">成功删除',
        job_id,
        '任务!</p>'
      )
    )
  } else if (status_code(r) == 500) {
    print_html(
      paste0(
        '<p style="color:red;">删除任务失败:',
        content(r)$error,
        '</p>'
      )
    )
  }
})

observe({
  req(job_list$list)
  .list <- job_list$list
  updateSelectInput(session, 'job_list_delete', choices = .list, selected = .list[1])
  updateSelectInput(session, 'ask_tell_box.job_id', choices = .list, selected = .list[1])
  updateSelectInput(session, 'feature_importance_box.job_id', choices = .list, selected = .list[1])
})

output$data_info <- renderDataTable({
  job_table()
}, options = list(
  pageLength = 10, scrollX = F, autoWidth = TRUE,
  columnDefs = list(list(width = '20px', targets = c(0, 1))))
)