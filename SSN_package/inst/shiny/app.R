

# Install CRAN packages
install.packages(c(
  "shiny",
  "readxl",
  "ggplot2",
  "data.table",
  "devtools",
  "matrixStats",
  "roxygen2"
))

# Install Bioconductor packages
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install(c(
  "phyloseq",
  "SpiecEasi",
  "microbiome"
))

# Install GitHub packages
devtools::install_github("zdk123/SpiecEasi")
devtools::install_github("gmteunisse/fantaxtic")
devtools::install("E:/Onedrive/labChenc/imeta/SSN2") 
# Install SSNR (if package source is available)
# devtools::install_local("path/to/SSNR.zip")  # For local installation
# devtools::install_github("yourusername/SSNR")  # If hosted on GitHub
library(shiny)
library(SSN2)
library(readxl)
library(ggplot2)
# 定义 UI
ui <- fluidPage(
  titlePanel("SSNR Shiny App"),
  sidebarLayout(
    sidebarPanel(
      fileInput("file1", "Choose .Rdata File", accept = c(".Rdata")),
      textInput("mypath", "Path", value = "E:/test/"),
      textInput("phylo_list", "Phylo List", value = 'aphy_test_resp.Rdata'),
      textInput("graph", "Graph Parameter", value = "graph"),
      actionButton("generate", "Generate SSN"),
      actionButton("extract", "Extract Attributes"),
      hr(),
      fileInput("excel_file", "Choose Excel File", accept = c(".xlsx", ".xls")),
      uiOutput("sheet_selector"),
      actionButton("plot", "Plot Data")
    ),
    mainPanel(
      tableOutput("contents"),
      tableOutput("attributes"),
      plotOutput("barplot")
    )
  )
)

# 定义服务器逻辑
server <- function(input, output, session) {
  observeEvent(input$generate, {
    req(input$file1)
    load(input$file1$datapath)
    phylo_list <- strsplit(input$phylo_list, ",")[[1]]
    generatSSN(input$mypath, phylo_list)
    
    # 动态生成的输出文件名
    phylotxt <- tools::file_path_sans_ext(basename(input$phylo_list))
    output_file <- paste0(input$mypath, phylotxt, "_comselSSN.csv")
    
    output$contents <- renderTable({
      data <- read.csv(output_file)
      head(data)
    })
  })
  
  observeEvent(input$extract, {
    req(input$file1)
    load(input$file1$datapath)
    g <- get(input$graph)
    attributes <- extr_netattr(g)
    
    output$attributes <- renderTable({
      attributes
    })
  })

  observe({
    req(input$excel_file)
    sheets <- excel_sheets(input$excel_file$datapath)
    updateSelectInput(session, "sheet", "Select Sheet", choices = sheets)
  })

  output$sheet_selector <- renderUI({
    req(input$excel_file)
    selectInput("sheet", "Select Sheet", choices = NULL)
  })

  observeEvent(input$plot, {
    req(input$excel_file, input$sheet)
    data <- read_excel(input$excel_file$datapath, sheet = input$sheet)
    data <- data[order(data$importance, decreasing = TRUE), ]
    
    output$barplot <- renderPlot({
      ggplot(data, aes(x = reorder(Feat, importance), y = importance, fill = importance)) +
        geom_bar(stat = "identity", show.legend = FALSE) +  # 隐藏图例
        geom_text(aes(label = importance), hjust = -0.3) +
        scale_fill_gradient(low = "red", high = "darkred") +
        coord_flip() +
        labs(title = "Feature Importance", x = "Feature", y = "Importance") +
        theme_void() +
        theme(
          axis.text.y = element_text(size = 12),  # 增大行名字字体
          axis.title = element_text(size = 14),   # 增大轴标题字体
          plot.title = element_text(size = 16, face = "bold")  # 增大标题字体并加粗
        )
    })
  })
}

# 运行应用
shinyApp(ui, server)