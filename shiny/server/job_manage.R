# NOTE: to trigger re-rendering of `job_table`
trigger_update_job_table <- reactiveVal()

search_space <- reactive({
  req(input$upload.add_zip$datapath)
  json_data <- jsonlite::read_json(input$upload.add_zip$datapath)
})

session$onRestore(
  function() {
    trigger_update_job_table(rnorm(1))
  }
)

# table of current job information
job_table <- eventReactive(
  trigger_update_job_table(), {
    r <- GET(
      address,
      query = list(check_job = 'null')
    )
    as.data.frame(do.call(rbind, content(r)))
  }
)

# create a job
observeEvent(input$upload.create_job, {
  .cs <- search_space()

  if (length(.cs) != 0) {
    r <- POST(address, body = .cs, encode = 'json')
    if (status_code(r) == 200) {
      job_id <- content(r)$job_id
      trigger_update_job_table(rnorm(1))

      print_html(
        paste0(
          '<p style="color:red;">成功创建',
          job_id,
          '任务!</p>'
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

# delete a job
observeEvent(input$upload.delete_job, {
  req(input$job_list_delete)
  job_id <- input$job_list_delete

  r <- GET(
    address,
    query = list(finalize = 'null', job_id = job_id)
  )

  if (status_code(r) == 200) {
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
  trigger_update_job_table(rnorm(1))
})

# update all control widgets depending on `job_table`
observe({
  req(job_table())
  .list <- rownames(job_table())

  if (length(.list) == 0)
    .selected <- NULL
  else
    .selected <- .list[1]

  updateSelectInput(session, 'job_list_delete', choices = .list, selected = .selected)
  updateSelectInput(session, 'ask_tell_box.job_id', choices = .list, selected = .selected)
  updateSelectInput(session, 'feature_importance_box.job_id', choices = .list, selected = .selected)
})

# handle the download request to the job table
output$data_info <- renderDataTable({
  job_table()
}, options = list(
  pageLength = 10, scrollX = F, autoWidth = TRUE,
  columnDefs = list(list(width = '20px', targets = c(0, 1))))
)

# handle the download request to the example JSON file
output$upload_box.download <- downloadHandler(
  filename = function() {
    'example.json'
  },
  content = function(.file) {
    data <- toJSON(
      read_json('example_continuous.json'),
      pretty = TRUE, auto_unbox = TRUE
    )
    con <- file(.file)
    writeLines(data, con)
    close (con)
  }
)