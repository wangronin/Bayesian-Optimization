upload_box <- function(width = 12, collapsible = T, collapsed = T, 
                       height = '300px') {  
  box(
    title = HTML('<p style="font-size:120%;">创建优化任务</p>'), 
    width = width, height = height, collapsed = collapsed, collapsible = collapsible, 
    solidHeader = T, status = "primary", 
    sidebarPanel(
      width = 12,
      fileInput(
        "upload.add_zip", 
        label = HTML('<p align="left">
                     Please choose a <i>JSON file</i> specifying the search space.</p>'),
        multiple = FALSE, accept = c("Application/json", ".json")
      ),

      actionButton(
        'upload.create_job', 
        label = HTML(
          '<p align="center" style="margin-bottom:0;"><b>生成优化任务实例</b></p>'
          )
        )
      )
    )
}

upload_prompt_box <- function(width = 12, collapsible = T, collapsed = T) {
  box(title = HTML('<p style="font-size:120%;">Data Processing Prompt</p>'),
      width = width, solidHeader = T, status = "primary",
      collapsible = collapsible, collapsed = collapsed,

      verbatimTextOutput('process_data_promt'),
      tags$head(
        tags$style(
          "#process_data_promt{
             color:black; font-size:12px; font-style:italic;
             max-height: 500px;
             overflow-y:visible; overflow-x: auto;
             white-space: pre-wrap;
             white-space: -moz-pre-wrap;
             white-space: -pre-wrap;
             white-space: -o-pre-wrap;
             word-wrap: normal;
             background: ghostwhite;
          }"
        )
      )
    )
}

job_list_box <- function(width = 12, collapsible = T, collapsed = T) {
  box(
    title = HTML('<p style="font-size:120%;">当前活跃任务</p>'),
    width = width, solidHeader = T, status = "primary",
    collapsible = collapsible, collapsed = collapsed,
    dataTableOutput('data_info'),
    
    selectInput(
      'job_list_delete', label = "请在如下列表中选择要删除的任务",
      choices = NULL, selected = NULL, width = '50%'
    ),
    actionButton(
      'upload.delete_job', 
      label = HTML(
        '<p align="center" style="margin-bottom:0;"><b>删除优化任务实例</b></p>'
      )
    )
  )
}

ask_box <- function(width = 12, collapsible = T, collapsed = T) {
  box(
    title = HTML('<p style="font-size:120%;">获取一组解</p>'),
    width = width, solidHeader = T, status = "primary",
    collapsible = collapsible, collapsed = collapsed,
    
    selectInput(
      'ask_box.job_id', label = "请在如下列表中选择任务ID",
      choices = NULL, selected = NULL, width = '50%'
    ),
    actionButton(
      'ask_box.ask', 
      label = HTML(
        '<p align="center" style="margin-bottom:0;"><b>获取一组解</b></p>'
      )
    ),
    
    dataTableOutput('ask_data'),
    downloadButton("ask_download", "下载这组解")
  )
}

tell_box <- function(width = 12, collapsible = T, collapsed = T) {
  box(
    title = HTML('<p style="font-size:120%;">向服务器返回目标值</p>'),
    width = width, solidHeader = T, status = "primary",
    collapsible = collapsible, collapsed = collapsed,

    selectInput(
      'tell_box.job_id', label = "请在如下列表中选择任务ID",
      choices = NULL, selected = NULL, width = '50%'
    ),
    fileInput(
      "tell_box.add_json", 
      label = HTML('<p align="left">Please choose a <i>CSV file</i> containing the target values</p>'),
      multiple = FALSE, accept = c("Application/json", ".json")
    ),
    
    actionButton(
      'tell_box.tell', 
      label = HTML(
        '<p align="center" style="margin-bottom:0;"><b>发送目标任务值</b></p>'
      )
    ),
    
    verbatimTextOutput('tell_data_promt'),
    tags$head(
      tags$style(
        "#tell_data_promt{
             color:black; font-size:12px; font-style:italic;
             max-height: 500px;
             overflow-y:visible; overflow-x: auto;
             white-space: pre-wrap;
             white-space: -moz-pre-wrap;
             white-space: -pre-wrap;
             white-space: -o-pre-wrap;
             word-wrap: normal;
             background: ghostwhite;
          }"
      )
    )
  )
}

progress_box <- function(width = 12, collapsible = T, collapsed = T) {
  box(title = HTML('<p style="font-size:120%;">任务进度</p>'),
      width = width, collapsible = collapsible, solidHeader = T,
      status = "primary", collapsed = collapsed,
      column(
        width = 12,
        align = "center",
        
        # HTML_P('The <b><i>mean, median, standard deviation and ERT</i></b> of the runtime       samples
        #       are depicted against the best objective values.
        #       The displayed elements (mean, median, standard deviations and ERT)
        #       can be switched on and off by clicking on the legend on the right.
        #       A <b>tooltip</b> and <b>toolbar</b> appears when hovering over the figure.'),
        plotlyOutput.IOHanalyzer('progress_box.line_chart')
      )
  )
}

feature_importance_box <- function(width = 12, collapsible = T, collapsed = T) {
  box(title = HTML('<p style="font-size:120%;">参数重要性</p>'),
      width = width, collapsible = collapsible, solidHeader = T,
      status = "primary", collapsed = collapsed,
        selectInput(
          'feature_importance_box.job_id', label = "请在如下列表中选择任务ID",
          choices = NULL, selected = NULL, width = '50%'
        ),
        # HTML_P('The <b><i>mean, median, standard deviation and ERT</i></b> of the runtime       samples
        #       are depicted against the best objective values.
        #       The displayed elements (mean, median, standard deviations and ERT)
        #       can be switched on and off by clicking on the legend on the right.
        #       A <b>tooltip</b> and <b>toolbar</b> appears when hovering over the figure.'),
        plotlyOutput.IOHanalyzer('feature_importance_box.bar_chart')
  )
}
