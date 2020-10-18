upload_box <- function(width = 12, collapsible = T, collapsed = T,
                       height = '400px') {
  box(
    title = HTML('<p style="font-size:120%;">创建优化任务</p>'),
    width = width, height = height, collapsed = collapsed, collapsible = collapsible,
    solidHeader = T, status = "primary",
    sidebarPanel(
      width = 12,
      
      downloadButton("upload_box.download", "下载JSON配置文件模板"),
      hr(),
      
      fileInput(
        "upload.add_zip",
        label = HTML(
          '<p align="left">
           请上传一个<i>JSON配置文件</i>来创建优化实例</p>'
        ),
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
  box(
    title = HTML('<p style="font-size:120%;">Data Processing Prompt</p>'),
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

ask_tell_box <- function(width = 12, collapsible = T, collapsed = T) {
  box(
    title = HTML('<p style="font-size:120%;">Ask-Tell接口</p>'),
    width = width, solidHeader = T, status = "primary",
    collapsible = collapsible, collapsed = collapsed,
    
    fluidRow(
      column( 
        width = 4, align = "left", offset = 0, 
        style = 'padding-left:15px; padding-right:15px; padding-top:5px; padding-bottom:5px',
        selectInput(
          'ask_tell_box.job_id', label = "请在如下列表中选择任务ID",
          choices = NULL, selected = NULL, width = '50%'
        ),
        actionButton(
          'ask_tell_box.ask',
          label = HTML(
            '<p align="center" style="margin-bottom:0;"><b>获取一组解/Ask</b></p>'
          )
        )
      ),
      
      column(
        width = 8, align = "left", offset = 0, 
        style = 'padding-left:15px; padding-right:15px; padding-top:5px; padding-bottom:5px',
        verbatimTextOutput('tell_data_promt'),
        tags$head(
          tags$style(
            "#tell_data_promt{
               color:black; font-size:12px; font-style:italic;
               max-height: 130px;
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
    ),
    
    hr(),
    br(),
    
    fluidRow(
      column(
        width = 8, align = "center",
        HTML_P(
          '您可以直接编辑如下的表格来添加目标值，
           并在结束添加之后点击“发送目标任务值”按钮来向服务器告知所添加的目标值'
        ),
        DT::dataTableOutput('ask_tell_box.data_table')
      ),
      
      column(
        width = 4, align = "left",
        actionButton(
          'ask_tell_box.tell',
          label = HTML(
            '<p align="center" style="margin-bottom:0;"><b>发送目标任务值/Tell</b></p>'
          )
        ),
        
        hr(),
        HTML_P('您也可以下载这组解(JSON格式)，并向其中加入一个名为`y`的字段来包含目标值。
           在您完成编辑后，可以直接上传这个JSON文件，之后上面的数据表格会自动更新。'),
        downloadButton("ask_tell_box.download", "您也可以下载这组解"),
        
        br(),
        br(),
        br(),
        
        fileInput(
          "ask_tell_box.add_json",
          label = HTML('<p align="left">上传编辑后的<i>JSON</i>文件</p>'),
          multiple = FALSE, accept = c("Application/json", ".json")
        )
      )
    )
  )
}

tell_box <- function(width = 12, collapsible = T, collapsed = T) {
  box(
    title = HTML('<p style="font-size:120%;">向服务器返回目标值</p>'),
    width = width, solidHeader = T, status = "primary",
    collapsible = collapsible, collapsed = collapsed,

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
  box(
    title = HTML('<p style="font-size:120%;">任务进度</p>'),
    width = width, collapsible = collapsible, solidHeader = T,
    status = "primary", collapsed = collapsed,
    column(
      width = 12,
      align = "center",
      .plotlyOutput('progress_box.line_chart')
    )
  )
}

feature_importance_box <- function(width = 12, collapsible = T, collapsed = T) {
  box(
    title = HTML('<p style="font-size:120%;">参数重要性</p>'),
    width = width, collapsible = collapsible, solidHeader = T,
    status = "primary", collapsed = collapsed,
    selectInput(
      'feature_importance_box.job_id', label = "请在如下列表中选择任务ID",
      choices = NULL, selected = NULL, width = '50%'
    ),
    .plotlyOutput('feature_importance_box.bar_chart')
  )
}
