FCE_GRID_INPUT_TEXT <- '<p align="justify">Set the range and the granularity of the results.
                        The table will show function values that have been reached within evenly
                        spaced evaluation budgets.</p>'

RT_TAR_LABEL <- HTML('<p>\\(B:\\) Budget value</p>')
RT_MIN_LABEL <- HTML('<p>\\(B_{\\text{min}}:\\) Smallest budget value</p>')
RT_MAX_LABEL <- HTML('<p>\\(B_{\\text{max}}:\\) Largest budget value</p>')
RT_STEP_LABEL <- HTML('<p>\\(\\Delta B:\\) Granularity (step size)</p>')

F_TAR_LABEL <- HTML('Target function value <p>\\(f_{\\text{target}}:\\)</p>')
F_MIN_LABEL <- HTML('<p>\\(f_{\\text{min}}:\\) Smallest target value</p>')
F_MAX_LABEL <- HTML('<p>\\(f_{\\text{max}}:\\) Largest target value</p>')
F_STEP_LABEL <- HTML('<p>\\(\\Delta f:\\) Granularity (step size)</p>')

Pdsc_info <- "Practical Deep Comparison uses this 
                as a threshold parameter to determine when two performance measure values are 
                equal or different from a practical point of view.
                Since the practical significance 
                is handled with preprocessing using the predefined threshold in sequential order of 
                the obtained runs, to make more robust analysis a Monte-Carlo simulation is required."

Pdsc_mc_info <- "Practical Deep Comparison uses this 
                to determine the number of monte-carlo simulations to perform. 
                The Monte-Carlo simulation involves preprocessing done with different orders of the 
                independent runs before comparing the distribution of the data.
                This only has effect if threshold for practical significance > 0"

HTML_P <- function(s) HTML(paste0('<p align="left" style="font-size:120%;">', s, '</p>'))

alg_select_info <- "Use this option to select which algorithms to plot. 
      This will hava an effect the dowloaded plot, 
      as opposed to using the legend-entries to show or hide algorithms "

custom_icon <- function(name = NULL){
  if (is.null(name)) name <- "info-circle"
  htmltools::tags$a(shiny::icon(name = name), href = "javascript:void(0);")
}