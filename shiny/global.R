suppressMessages(library(shiny))
suppressMessages(library(shinyjs))
suppressMessages(library(magrittr))
suppressMessages(library(data.table))
suppressMessages(library(plotly))
suppressMessages(library(shinydashboard))
suppressMessages(library(xtable))
suppressMessages(library(colourpicker))
suppressMessages(library(bsplus))
suppressMessages(library(DT))
suppressMessages(library(knitr))
suppressMessages(library(kableExtra))


line_types <- c("solid", "dot", "dash", "longdash", "dashdot", "longdashdot")


Set1 <- function(n) colorspace::sequential_hcl(n, h = c(360, 40), c. = c(100, NA, 90), l = c(28, 90),
                                   power = c(1, 1.1), gamma = NULL, fixup = TRUE,
                                   alpha = 1)#, palette = NULL, rev = FALSE)

Set2 <- function(n) colorspace::sequential_hcl(n, c(261, 26), c. = c(50, NA, 70), l = c(54, 77),
                                   power = c(0.5, NA), gamma = NULL,
                                   fixup = TRUE, alpha = 1)#, palette = NULL, rev = FALSE)

Set3 <- function(n) colorspace::sequential_hcl(n, c(-88, 59), c. = c(60, 75, 55), l = c(40, 90),
                                   power = c(0.1, 1.2), gamma = NULL,
                                   fixup = TRUE, alpha = 1)#, palette = NULL, rev = FALSE)


color_palettes <- function(ncolor) {
  max_colors <- 2
  if (ncolor <= max_colors) return(Set3)
  
  brewer <- function(n) {
    colors <- RColorBrewer::brewer.pal(n, 'Spectral')
    colors[colors == "#FFFFBF"] <- "#B2B285"
    colors[colors == "#E6F598"] <- "#86FF33"
    colors[colors == '#FEE08B'] <- "#FFFF33"
    colors
  }
  
  color_fcts <- c(colorRamps::primary.colors, Set3)
  
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

# font No. 1...
f1 <- list(
  family = 'Old Standard TT, serif',
  size = 11,
  color = 'black'
)

# font No. 2...
f2 <- list(
  family = 'Old Standard TT, serif',
  size = 13,
  color = 'black'
)

# font No. 3...
f3 <- function() {
  list(
    family = 'Old Standard TT, serif',
    size = getOption("IOHanalyzer.tick_fontsize", default = 12), 
    color = 'black'
  )
}

legend_right <- function() {
  list(x = 1.01, y = 1, orientation = 'v',
       font = list(size = getOption("IOHanalyzer.legend_fontsize", default = 18), 
                   family = 'Old Standard TT, serif'))
}

legend_inside <- function() {
  list(x = .01, y = 1, orientation = 'v',
       bgcolor = 'rgba(255, 255, 255, 0)',
       bordercolor = 'rgba(255, 255, 255, 0)',
       font = list(size = getOption("IOHanalyzer.legend_fontsize", default = 18), 
                   family = 'Old Standard TT, serif'))
}

legend_inside2 <- function() { 
  list(x = 0.7, y = 0.1, orientation = 'v',
       bgcolor = 'rgba(255, 255, 255, 0.5)',
       bordercolor = 'rgba(255, 255, 255, 0.8)',
       font = list(size = getOption("IOHanalyzer.legend_fontsize", default = 18), 
                   family = 'Old Standard TT, serif'))
}

legend_below <- function() { 
  list(y = -0.2, orientation = 'h',
       font = list(size = getOption("IOHanalyzer.legend_fontsize", default = 18), 
                   family = 'Old Standard TT, serif'))
}

legend_location <- function(){
  opt <- getOption('IOHanalyzer.legend_location', default = 'below')
  if (opt == 'outside_right') return(legend_right())
  else if (opt == 'inside_left') return(legend_inside())
  else if (opt == 'inside_right') return(legend_inside2())
  else if (opt == 'below') return(legend_below())
  # else if (opt == 'below2') return(legend_below2())
  else warning(paste0("The selected legend option (", opt, ") is not implemented"))
}

# TODO: create font object as above for title, axis...

#' Template for creating plots in the IOHanalyzer-style
#' 
#' @param title Title for the plot
#' @param x.title X-axis label
#' @param y.title Y-axis label
#' 
#' @export
#' @examples 
#' IOH_plot_ly_default("Example plot","x-axis","y-axis") 
IOH_plot_ly_default <- function(title = NULL, x.title = NULL, y.title = NULL) {
  plot_ly() %>%
    layout(title = list(text = title, 
                        font = list(size = getOption("IOHanalyzer.title_fontsize", default = 16),
                                    family = 'Old Standard TT, serif')),
           autosize = T, hovermode = 'compare',
           legend = legend_location(),
           paper_bgcolor = 'rgb(255,255,255)',
           plot_bgcolor = getOption('IOHanalyzer.bgcolor'),
           font = list(size = getOption("IOHanalyzer.label_fontsize", default = 16),
                       family = 'Old Standard TT, serif'),
           autosize = T,
           showlegend = T, 
           xaxis = list(
             # title = list(text = x.title, font = f3),
             title = x.title,
             gridcolor = getOption('IOHanalyzer.gridcolor'),
             showgrid = TRUE,
             showline = FALSE,
             showticklabels = TRUE,
             tickcolor = getOption('IOHanalyzer.tickcolor'),
             ticks = 'outside',
             ticklen = 9,
             tickfont = f3(),
             exponentformat = 'e',
             zeroline = F),
           yaxis = list(
             # title = list(text = y.title, font = f3),
             title = y.title,
             gridcolor = getOption('IOHanalyzer.gridcolor'),
             showgrid = TRUE,
             showline = FALSE,
             showticklabels = TRUE,
             tickcolor = getOption('IOHanalyzer.tickcolor'),
             ticks = 'outside',
             ticklen = 9,
             tickfont = f3(),
             exponentformat = 'e',
             zeroline = F))
}

# global options
options(datatable.print.nrows = 20)
options(width = 80)
options(shiny.maxRequestSize = 200 * 1024 ^ 2)  # maximal upload file size

# for customized 'plotlyOutput' function -----
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

plotlyOutput.IOHanalyzer <- function(outputId, width = '100%', aspect_ratio = 16/10) {
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

# markers for plotly
symbols <- c("circle-open", "diamond-open", "square-open", "cross-open",
             "triangle-up-open", "triangle-down-open")

# ploting settings for UI ---------------------
plotly_height <- "auto"
plotly_width <- "auto"
plotly_height2 <- "auto"
plotly_width2 <- "auto"

print_html <- function(s, widget_id = 'process_data_promt') 
  shinyjs::html(widget_id, s, add = TRUE)

print_html2 <- function(s, widget_id = 'tell_data_promt') 
  shinyjs::html(widget_id, s, add = TRUE)


# download file names: csv, image ---------------------
overview_single_name <- parse(text = "paste0('Overview-', paste0(input$Overall.Dim, 'D'),
                             paste0('F', input$Overall.Funcid), '.', input$Overview.Single.Format)")
overview_all_name <- parse(text = "paste0('Overview-All-', '.', input$Overview.All.Format)")
RT_csv_name <- parse(text = "paste0('RTStats-', paste0(input$Overall.Dim, 'D'),
                             paste0('F', input$Overall.Funcid), '.', input$RTSummary.Statistics.Format)")
RT_overview_name <- parse(text = "paste0('RTOverview-', paste0(input$Overall.Dim, 'D'),
                             paste0('F', input$Overall.Funcid), '.', input$RTSummary.Overview.Format)")
RTSample_csv_name <- parse(text = "paste0('RTSample-', paste0(input$Overall.Dim, 'D'),
                             paste0('F', input$Overall.Funcid), '.', input$RTSummary.Sample.Format)")
FV_csv_name <- parse(text = "paste0('FVStats-', paste0(input$Overall.Dim, 'D'),
                             paste0('F', input$Overall.Funcid), '.', input$FCESummary.Statistics.Format)")
FV_overview_name <- parse(text = "paste0('FVOverview-', paste0(input$Overall.Dim, 'D'),
                             paste0('F', input$Overall.Funcid), '.', input$FCESummary.Overview.Format)")
FVSample_csv_name <- parse(text = "paste0('FVSample-', paste0(input$Overall.Dim, 'D'),
                             paste0('F', input$Overall.Funcid), '.', input$FCESummary.Sample.FileFormat)")
FV_PAR_csv_name <- parse(text = "paste0('PARSummary-', paste0(input$Overall.Dim, 'D'),
                             paste0('F', input$Overall.Funcid), '.', input$FV_PAR.Summary.Format)")
FV_PARSample_csv_name <- parse(text = "paste0('PARSample-', paste0(input$Overall.Dim, 'D'),
                             paste0('F', input$Overall.Funcid), '.', input$FV_PAR.Sample.FileFormat)")
RT_PAR_csv_name <- parse(text = "paste0('PARSummary-', paste0(input$Overall.Dim, 'D'),
                             paste0('F', input$Overall.Funcid), '.', input$RT_PAR.Summary.Format)")
RT_PARSample_csv_name <- parse(text = "paste0('PARSample-', paste0(input$Overall.Dim, 'D'),
                             paste0('F', input$Overall.Funcid), '.', input$RT_PAR.Sample.FileFormat)")
ERT_multi_func_name <- parse(text = "paste0('MultiERT-', paste0(input$Overall.Dim, 'D'),
                             '.', input$ERTPlot.Aggr.TableFormat)")
ERT_multi_dim_name <- parse(text = "paste0('MultiERT-', paste0('F', input$Overall.Funcid),
                             '.', input$ERTPlot.Aggr_Dim.TableFormat)")
FCE_multi_func_name <- parse(text = "paste0('MultiFCE-', paste0(input$Overall.Dim, 'D'),
                             '.', input$FCEPlot.Aggr.TableFormat)")
RT_Glicko2_table_name <- parse(text = "paste0('RT_Glicko2', '.', input$RT_Stats.Glicko.TableFormat)")
RT_Glicko2_figure_name <- parse(text = "paste0('RT_Glicko2', '.', input$RT_Stats.Glicko.Format)")

RT_DSC_table_name <- parse(text = "paste0('RT_DSC', '.', input$RT_Stats.DSC.TableFormat)")
RT_DSC_figure_name <- parse(text = "paste0('RT_DSC', '.', input$RT_Stats.DSC.Format)")
RT_DSC_figure_name_rank <- parse(text = "paste0('RT_DSC_PerformViz', '.', input$RT_Stats.DSC.Format_rank)")
RT_DSC_table_name_rank <- parse(text = "paste0('RT_DSC_Rank', '.', input$RT_Stats.DSC.TableFormat_rank)")

FV_DSC_table_name <- parse(text = "paste0('FV_DSC', '.', input$FV_Stats.DSC.TableFormat)")
FV_DSC_figure_name <- parse(text = "paste0('FV_DSC', '.', input$FV_Stats.DSC.Format)")
FV_DSC_figure_name_rank <- parse(text = "paste0('FV_DSC_PerformViz', '.', input$FV_Stats.DSC.Format_rank)")
FV_DSC_table_name_rank <- parse(text = "paste0('FV_DSC_Rank', '.', input$FV_Stats.DSC.TableFormat_rank)")

RT_Stats_table_name <- parse(text = "paste0('RT_Stat_Comp-', paste0(input$Overall.Dim, 'D'),
                             paste0('F', input$Overall.Funcid), '.', input$RT_Stats.Overview.TableFormat)")
RT_Stats_heatmap_name <- parse(text = "paste0('RT_Stat_Heatmap-', paste0(input$Overall.Dim, 'D'),
                             paste0('F', input$Overall.Funcid), '.', input$RT_Stats.Overview.Format)")
RT_Stats_network_name <- parse(text = "paste0('RT_Stat_Network-', paste0(input$Overall.Dim, 'D'),
                             paste0('F', input$Overall.Funcid), '.', input$RT_Stats.Overview.Format)")
# max_samples <- 100

FIG_NAME_ERT_PER_FUN <- parse(text = "paste0('ERT-', Sys.Date(), '.', input$ERTPlot.Format)")
FIG_NAME_ERT_PER_FUN_MULTI <- parse(text = "paste0('ERT_Mult-', Sys.Date(), '.', input$ERTPlot.Multi.Format)")
FIG_NAME_ERT_AGGR <- parse(text = "paste0('ERT_Aggr-', Sys.Date(), '.', input$ERTPlot.Aggr.Format)")
FIG_NAME_ERT_AGGR_DIM <- parse(text = "paste0('ERT_Aggr_Dim-', Sys.Date(), '.', input$ERTPlot.Aggr_Dim.Format)")
FIG_NAME_RT_PMF <- parse(text = "paste0('RT_PMF-', Sys.Date(), '.', input$RTPMF.Bar.Format)")
FIG_NAME_RT_HIST <- parse(text = "paste0('RT_HIST-', Sys.Date(), '.', input$RTPMF.Hist.Format)")
FIG_NAME_RT_ECDF_AGGR <- parse(text = "paste0('RT_ECDF_AGGR-', Sys.Date(), '.', input$RTECDF.Multi.Format)")
FIG_NAME_RT_ECDF_MULT <- parse(text = "paste0('RT_ECDF_MULT-', Sys.Date(), '.', input$RTECDF.Aggr.Format)")
FIG_NAME_RT_AUC <- parse(text = "paste0('RT_AUC-', Sys.Date(), '.', input$RTECDF.AUC.Format)")

FIG_NAME_FV_PER_FUN <- parse(text = "paste0('FV-', Sys.Date(), '.', input$FCEPlot.Format)")
FIG_NAME_FV_PER_FUN_MULTI <- parse(text = "paste0('FCE_Mult-', Sys.Date(), '.', input$FCEPlot.Multi.Format)")
FIG_NAME_FV_AGGR <- parse(text = "paste0('FCE_Aggr-', Sys.Date(), '.', input$FCEPlot.Aggr.Format)")
FIG_NAME_FV_PDF <- parse(text = "paste0('FV_PMF-', Sys.Date(), '.', input$FCEPDF.Bar.Format)")
FIG_NAME_FV_HIST <- parse(text = "paste0('FV_HIST-', Sys.Date(), '.', input$FCEPDF.Hist.Format)")
FIG_NAME_FV_ECDF_AGGR <- parse(text = "paste0('FV_ECDF_AGGR-', Sys.Date(), '.', input$FCEECDF.Mult.Format)")
FIG_NAME_FV_AUC <- parse(text = "paste0('FV_AUC-', Sys.Date(), '.', input$FCEECDF.AUC.Format)")

FIG_NAME_RT_PAR_PER_FUN <- parse(text = "paste0('RT_PAR-', Sys.Date(), '.', input$RT_PAR.Plot.Format)")
FIG_NAME_FV_PAR_PER_FUN <- parse(text = "paste0('FV_PAR-', Sys.Date(), '.', input$FV_PAR.Plot.Format)")


# ID of the control widget, whose current value should de always recorded and restored ----
# those control widget are switched on and off
widget_id <- c('RTSummary.Statistics.Min',
               'RTSummary.Statistics.Max',
               'RTSummary.Statistics.Step',
               'RTSummary.Sample.Min',
               'RTSummary.Sample.Max',
               'RTSummary.Sample.Step',
               'RTECDF.Multi.Min',
               'RTECDF.Multi.Max',
               'RTECDF.Multi.Step',
               'RTECDF.Single.Target',
               'RTPMF.Bar.Target',
               'RTPMF.Hist.Target',
               'ERTPlot.Min',
               'ERTPlot.Max',
               'ERTPlot.Aggr.Targets',
               'RTECDF.AUC.Min',
               'RTECDF.AUC.Max',
               'RTECDF.AUC.Step',
               'FV_PAR.Plot.Min',
               'FV_PAR.Plot.Max',
               'FV_PAR.Summary.Min',
               'FV_PAR.Summary.Max',
               'FV_PAR.Summary.Step',
               'FV_PAR.Sample.Min',
               'FV_PAR.Sample.Max',
               'FV_PAR.Sample.Step',
               'RT_PAR.Plot.Min',
               'RT_PAR.Plot.Max',
               'RT_PAR.Summary.Min',
               'RT_PAR.Summary.Max',
               'RT_PAR.Summary.Step',
               'RT_PAR.Sample.Min',
               'RT_PAR.Sample.Max',
               'RT_PAR.Sample.Step',
               'FCESummary.Statistics.Min',
               'FCESummary.Statistics.Max',
               'FCESummary.Statistics.Step',
               'FCESummary.Sample.Min',
               'FCESummary.Sample.Max',
               'FCESummary.Sample.Step',
               'FCEPDF.Hist.Runtime',
               'FCEPDF.Bar.Runtime',
               'FCEPlot.Min',
               'FCEPlot.Max',
               'FCEECDF.Mult.Min',
               'FCEECDF.Mult.Max',
               'FCEECDF.Mult.Step',
               'FCEECDF.AUC.Min',
               'FCEECDF.AUC.Max',
               'FCEECDF.AUC.Step',
               'FCEECDF.Single.Target')

eventExpr <- parse(text = paste0('{', paste(paste0('input$', widget_id), collapse = "\n"), '}'))

# token needed for mapbox, which is again needed for ocra... ------
supported_fig_format <- c('png', 'eps', 'svg', 'pdf')
Sys.setenv('MAPBOX_TOKEN' = 'pk.eyJ1Ijoid2FuZ3JvbmluIiwiYSI6ImNqcmIzemhvMDBudnYzeWxoejh5c2Y5cXkifQ.9XGMWTDOsgi3-b5qG594kQ')

sanity_check_id <- function(input) {
  for (id in widget_id) {
    tryCatch(eval(parse(text = paste0('input$', id))),
             error = function(e) {
               cat(paste('widget', id, 'does not exist!\n'))
             })
  }
}






