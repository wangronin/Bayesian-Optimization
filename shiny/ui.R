for (f in list.files('ui', pattern = '.R', full.names = T)) {
  source(f)
}

header <- dashboardHeader(title = HTML('<div align="center"><b>AI Excelsior</b></div>'))

# The side bar layout ---------------------------------------------
sidebar <- dashboardSidebar(
  useShinyjs(),
  sidebar_menu()
)

body <- dashboardBody(
  tags$style(
    HTML('
       .popover-title {color:black;}
       .popover-content {color:black;}
       .main-sidebar {z-index:auto;}
       .fa-exclamation-triangle {color:#E87722}
       .sticky {
         position: fixed;
         top: 0;
         width: 100%;
       }

       .sticky2 {
         position: fixed;
       }

       .table {
          border-collapse: collapse;
          width: 100%;
        }
       .table td, tr {
          padding: 0px;
          margin: 0px;
          vertical-align: middle;
          height: 45px;
        }
       .table th {height: 0px;}'
    )
  ),

  # to show text on the header (heading banner)
  tags$head(
    tags$style(
      HTML('
        .myClass {
          font-size: 20px;
          line-height: 50px;
          text-align: left;
          font-family: "Helvetica Neue",Helvetica,Arial,sans-serif;
          padding: 0 15px;
          overflow: hidden;
          color: white;
        }'
      )
    )
  ),

  tags$head(
    tags$style(
      HTML(
        '.box-title {
            font-size: 20px;
            line-height: 50px;
            text-align: left;
            font-family: "Helvetica Neue",Helvetica,Arial,sans-serif;
            padding: 0 15px;
            overflow: hidden;
            color: white;
          }'
      )
    )
  ),

  tags$head(
    tags$style(
      HTML(
        "label { font-size:120%; }"
      )
    )
  ),

  tags$script(
    HTML('
      $(document).ready(function() {
        $("header").find("nav").append(\'<span class="myClass">Performance Evaluation for Iterative
        Optimization Heuristics</span>\');
      })
   ')
  ),

  tags$script(
    "Shiny.addCustomMessageHandler('background-color',
      function(color) {
        document.body.style.backgroundColor = color;
        document.body.innerText = color;
      });"
  ),

  tags$script(
    HTML('
      window.setInterval(function() {
        var elem = document.getElementById("process_data_promt");
        if (typeof elem !== "undefined" && elem !== null) elem.scrollTop = elem.scrollHeight;
      }, 20);'
    )
  ),

  tags$head(
    tags$script(
      HTML("
        Shiny.addCustomMessageHandler('manipulateMenuItem', function(message){
          var aNodeList = document.getElementsByTagName('a');

          for (var i = 0; i < aNodeList.length; i++) {
            if(aNodeList[i].getAttribute('data-value') == message.tabName || aNodeList[i].getAttribute('href') == message.tabName) {
              if(message.action == 'hide'){
                aNodeList[i].setAttribute('style', 'display: none;');
              } else {
                aNodeList[i].setAttribute('style', 'display: block;');
              };
            };
          }
        });
      ")
    )
  ),

  # make the data uploading prompt always scroll to the bottom
  tags$script(
    HTML('
       window.setInterval(function() {
         var elem = document.getElementById("upload_data_promt");
         if (typeof elem !== "undefined" && elem !== null) elem.scrollTop = elem.scrollHeight;
       }, 20);
    ')
  ),

  # render the header and the side bar 'sticky'
  tags$script(
    HTML(
      '// When the user scrolls the page, execute myFunction
      window.onscroll = function() {myFunction()};

      // Get the header
      var header = document.getElementById("header");

      // Get the side bar
      var sideBar = document.getElementById("sidebarCollapsed");
      sideBar.classList.add("sticky2");

      // Get the offset position of the navbar
      var sticky = header.offsetTop;

      // Add the sticky class to the header when you reach its scroll position.
      // Remove "sticky" when you leave the scroll position
      function myFunction() {
        if (window.pageYOffset > sticky) {
          header.classList.add("sticky");
        } else {
          header.classList.remove("sticky");
        }
      }'
    )
  ),

  # load MathJax
  # TODO: download MathJax and its license and include it in our package
  HTML("<head><script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML'
       async></script></head>"),
  # use_bs_tooltip(),
  # use_bs_popover(),

  tabItems(
    tabItem(
      tabName = 'upload',
      fluidRow(
        column(
          width = 6,
          upload_box(collapsible = F)
        ),

        column(
          width = 6,
          job_list_box(collapsible = F)
        )
      ),

      fluidRow(
        column(
          width = 6,
          upload_prompt_box(collapsible = F)
        )
      )
    ),

    tabItem(
      tabName = 'execute_job_panel',
      fluidRow(
        column(
          width = 12,
          ask_tell_box(collapsible = F)
        )
      ),

      fluidRow(
        column(
          width = 6,
          progress_box(collapsible = F)
        ),
        column(
          width = 6,
          feature_importance_box(collapsible = F)
        )
      )
    ),

    tabItem(tabName = 'documentation', includeMarkdown('USAGE.md'))
  )
)

# -----------------------------------------------------------
dashboardPage(title = 'AIxcelsior', header, sidebar, body)