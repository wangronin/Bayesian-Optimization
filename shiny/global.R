suppressMessages(library(shiny))
suppressMessages(library(shinyjs))
suppressMessages(library(magrittr))
suppressMessages(library(data.table))
suppressMessages(library(plotly))
suppressMessages(library(shinydashboard))
suppressMessages(library(DT))
suppressMessages(library(httr))
suppressMessages(library(jsonlite))

ip <- "106.14.217.207"
# ip <- '127.0.0.1'
port <- 7200
address <- paste0('http://', ip, ':', port)

HTML_P <- function(s) HTML(paste0('<p align="left" style="font-size:120%;">', s, '</p>'))

f1 <- function(size = 16) {
  list(
    size = size,
    family = 'Old Standard TT, serif'
  )
}

f2 <- function(size = 12) {
  list(
    family = 'Old Standard TT, serif',
    size = size,
    color = 'black'
  )
}

legend_below <- list(
  y = -0.2,
  orientation = 'h',
  font = f1()
)

.axis.style <- list(
  showgrid = TRUE,
  showline = FALSE,
  showticklabels = TRUE,
  ticks = 'outside',
  ticklen = 9,
  tickfont = f2(),
  exponentformat = 'e',
  zeroline = F
)

color_palettes <- function(ncolor) {
  brewer <- function(n) {
    colors <- RColorBrewer::brewer.pal(n, 'Spectral')
    colors[colors == "#FFFFBF"] <- "#B2B285"
    colors[colors == "#E6F598"] <- "#86FF33"
    colors[colors == '#FEE08B'] <- "#FFFF33"
    colors
  }

  color_fcts <- c(colorRamps::primary.colors)

  n <- min(11, ncolor)
  colors <- brewer(n)
  ncolor <- ncolor - n

  i <- 1
  while (ncolor > 0) {
    n <- min(8, ncolor)
    if (i > length(color_fcts)) {
      colors <- c(colors, colorRamps::primary.colors(ncolor))
      break
    } else {
      colors <- c(colors, color_fcts[[i]](n))
      ncolor <- ncolor - n
    }
    i <- i + 1
  }
  colors
}

.colors <- color_palettes(8)

plotly_default <- function(title = NULL, x.title = NULL, y.title = NULL) {
  plot_ly() %>%
    layout(
      title = list(text = title, font = f1()),
      autosize = T,
      hovermode = 'compare',
      legend = legend_below,
      paper_bgcolor = 'rgb(255,255,255)',
      font = f1(),
      autosize = T,
      showlegend = T,
      xaxis = c(list(title = x.title), .axis.style),
      yaxis = c(list(title = y.title), .axis.style)
    )
}

# global options
options(datatable.print.nrows = 20)
options(width = 80)
options(shiny.maxRequestSize = 200 * 1024 ^ 2)  # maximal upload file size

widget_html <- function(name, package, id, style, class, inline = FALSE, ...) {
  # attempt to lookup custom html function for widget
  fn <- tryCatch(get(paste0(name, "_html"),
                     asNamespace(package),
                     inherits = FALSE),
                 error = function(e) NULL)

  # call the custom function if we have one, otherwise create a div
  if (is.function(fn)) {
    fn(id = id, style = style, class = class, ...)
  } else if (inline) {
    tags$span(id = id, style = style, class = class)
  } else {
    tags$div(id = id, style = style, class = class)
  }
}

checkShinyVersion <- function(error = TRUE) {
  x <- utils::packageDescription('htmlwidgets', fields = 'Enhances')
  r <- '^.*?shiny \\(>= ([0-9.]+)\\).*$'
  if (is.na(x) || length(grep(r, x)) == 0 || system.file(package = 'shiny') == '')
    return()
  v <- gsub(r, '\\1', x)
  f <- if (error) stop else packageStartupMessage
  if (utils::packageVersion('shiny') < v)
    f("Please upgrade the 'shiny' package to (at least) version ", v)
}

widget_dependencies <- function(name, package){
  htmlwidgets::getDependency(name, package)
}

.plotlyOutput <- function(outputId, width = '100%', aspect_ratio = 16/10) {
  padding_bottom <- paste0(100 / aspect_ratio, '%')
  reportSize <- TRUE
  inline <- FALSE

  checkShinyVersion()
  html <- htmltools::tagList(
    widget_html('plotly', 'plotly', id = outputId,
                class = paste0('plotly', " html-widget html-widget-output",
                               if (reportSize)
                                 " shiny-report-size"),
                style = sprintf("width:%s; height: 0; padding-bottom:%s; %s",
                                htmltools::validateCssUnit(width),
                                htmltools::validateCssUnit(padding_bottom),
                                if (inline)
                                  "display: inline-block;"
                                else ""),
                width = width, height = 0)
  )
  dependencies <- widget_dependencies('plotly', 'plotly')
  htmltools::attachDependencies(html, dependencies)
}

print_html <- function(s, widget_id = 'process_data_promt')
  shinyjs::html(widget_id, s, add = TRUE)

print_html2 <- function(s, widget_id = 'tell_data_promt')
  shinyjs::html(widget_id, s, add = TRUE)