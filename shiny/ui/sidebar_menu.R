sidebar_menu <- function() {
  sidebarMenu(
    id = "tabs",
    menuItem(
      "创建任务", tabName = "upload",
      icon = icon('upload', lib = 'glyphicon'), selected = T
    ),
    menuItem(
      "优化任务面板", tabName = "execute_job_panel", icon = icon("line-chart")
    ),
    menuItem(
      "说明文档", tabName = "documentation", icon = icon("line-chart")
    )
  )
}
