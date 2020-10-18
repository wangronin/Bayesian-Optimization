tell_done <- reactiveValues(value = 0)

proxy <- dataTableProxy('ask_tell_box.data_table')
data_table <- reactiveVal(NULL)
trigger_renderDT <- reactiveVal(NULL)

observeEvent(input$ask_tell_box.ask, {
  job_id <- input$ask_tell_box.job_id
  r <- GET(
    address,
    query = list(ask = 'null', job_id = job_id)
  )

  if (status_code(r) == 200) {
    dt <- as.data.table(do.call(rbind, content(r)['X'][[1]]))
    dt$y <- 0
    data_table(dt)
    trigger_renderDT(rnorm(1))

    print_html2(
      paste0(
        '<p style="color:red;">任务',
        job_id,
        '获取一组解成功！</p>'
      )
    )
  } else if (status_code(r) == 500) {
    print_html2(
      paste0(
        '<p style="color:red;">获取一组解失败:',
        content(r)$error,
        '</p>'
      )
    )
  }
})

observeEvent(input$ask_tell_box.add_json, {
  path <- input$ask_tell_box.add_json$datapath
  req(path)

  json_data <- jsonlite::read_json(path)
  df <- as.data.table(do.call(rbind, json_data['X'][[1]]))
  df$y <- json_data$y

  if (is.null(data_table())) {
    data_table(df)
    trigger_renderDT(rnorm(1))
  } else if (ncol(df) == ncol(data_table())) {
    data_table(df)
    replaceData(
      proxy, data_table(),
      resetPaging = FALSE,
      rownames = FALSE
    )
  } else {
    data_table(df)
    trigger_renderDT(rnorm(1))
  }
})

observeEvent(input$ask_tell_box.data_table_cell_edit, {
  info <- input$ask_tell_box.data_table_cell_edit
  i <- info$row
  j <- info$col + 1
  v <- info$value

  suppressWarnings({
    dt <- data_table()
    set(dt, i, j, as.numeric(v))
    data_table(dt)
  })

  replaceData(
    proxy, data_table(),
    resetPaging = FALSE, rownames = FALSE
  )
})

observeEvent(input$ask_tell_box.tell, {
  job_id <- input$ask_tell_box.job_id

  dt <- data_table()
  X <- dt[, -c('y')]
  y <- dt$y
  json_data <- jsonlite::toJSON(list(X = X, y = y, job_id = job_id), auto_unbox = T)

  r <- POST(
    address,
    body = json_data, encode = 'json'
  )

  if (status_code(r) == 200) {
    print_html2(
      paste(
        '<p style="color:red;">成功添加目标值',
        paste(y, collapse = ', '),
        '至任务',
        job_id,
        '</p>'
      )
    )
    tell_done$value <- rnorm(1)
  } else if (status_code(r) == 500) {
    print_html2(
      paste0(
        '<p style="color:red;">添加目标值失败!',
        content(r)$error,
        '</p>'
      )
    )
  }
})

output$ask_tell_box.data_table <- DT::renderDataTable({
    trigger_renderDT()
    isolate({data_table()})
  },
  editable = TRUE,
  rownames = F,
  options = list(
    pageLength = 10,
    scrollX = T,
    autoWidth = TRUE,
    columnDefs = list(
      list(
        className = 'dt-right', targets = "_all"
      )
    )
  )
)

output$ask_tell_box.download <- downloadHandler(
  filename = function() {
    'solutions.json'
  },
  content = function(file) {
    dt <- data_table()
    X <- dt[, -c('y')]
    y <- dt$y
    jsonlite::write_json(list(X = X, y = y), file, auto_unbox = T)
  }
)