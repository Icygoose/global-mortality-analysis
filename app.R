# Load necessary libraries
library(shiny)
library(ggplot2)
library(plotly)
library(dplyr)
library(factoextra)
library(reshape2)
library(heatmaply) # For interactive correlation heatmaps
library(viridis)   # For enhanced color palettes

# Load your datasets and models
# Replace the following lines with your actual data and model loading code
# Example:
# model_df <- read.csv("path_to_your_model_data.csv")
# data_clean <- read.csv("path_to_your_clean_data.csv") # For EDA
# clustering_data <- read.csv("path_to_your_clustering_data.csv") # Ensure this is standardized
# tree_model <- readRDS("path_to_your_tree_model.rds")
# logistic_model <- readRDS("path_to_your_logistic_model.rds")

# Define UI
ui <- navbarPage(
  "Mortality Rate Analysis App",
  
  # Exploratory Data Analysis Tab
  tabPanel("Exploratory Data Analysis",
           sidebarLayout(
             sidebarPanel(
               selectInput("eda_plot_type", "Select Plot Type:",
                           choices = c("Single Variable", "Multiple Variables", "Correlation Heatmap", "Time Series")),
               conditionalPanel(
                 condition = "input.eda_plot_type == 'Single Variable'",
                 selectInput("eda_single_var", "Choose a Variable:",
                             choices = c("Outdoor.air.pollution", "High.systolic.blood.pressure", "Smoking", "Diet.high.in.sodium"))
               ),
               conditionalPanel(
                 condition = "input.eda_plot_type == 'Multiple Variables'",
                 selectInput("eda_x_var", "Choose X Variable:",
                             choices = c("Outdoor.air.pollution", "High.systolic.blood.pressure", "Smoking", "Diet.high.in.sodium")),
                 selectInput("eda_y_var", "Choose Y Variable:",
                             choices = c("Outdoor.air.pollution", "High.systolic.blood.pressure", "Smoking", "Diet.high.in.sodium"))
               ),
               conditionalPanel(
                 condition = "input.eda_plot_type == 'Time Series'",
                 selectInput("eda_time_var", "Choose a Variable:",
                             choices = c("Outdoor.air.pollution", "High.systolic.blood.pressure", "Smoking"))
               )
             ),
             mainPanel(
               tabsetPanel(
                 tabPanel("Plot", 
                          # Use conditionalPanel to switch between plot types
                          uiOutput("edaPlotUI")),
                 tabPanel("Summary", verbatimTextOutput("edaSummary"))
               )
             )
           )
  ),
  
  # Single Variable Model Performance Tab
  tabPanel("Single Variable Performance",
           sidebarLayout(
             sidebarPanel(
               selectInput("variable", "Choose a Variable:",
                           choices = c("GDP_per_Capita", "Diet.low.in.fruits", "Alochol.use", "Diet.high.in.sodium"))
             ),
             mainPanel(
               tabsetPanel(
                 tabPanel("Histogram", plotlyOutput("singleVarHist")),
                 tabPanel("Boxplot", plotlyOutput("singleVarBox")),
                 tabPanel("Summary", verbatimTextOutput("singleVarSummary"))
               )
             )
           )
  ),
  
  # Classification Models Performance Tab
  tabPanel("Classification Models",
           sidebarLayout(
             sidebarPanel(
               checkboxGroupInput("models", "Select Models to Compare:",
                                  choices = c("Decision Tree", "Logistic Regression"),
                                  selected = c("Decision Tree", "Logistic Regression"))
             ),
             mainPanel(
               tabsetPanel(
                 tabPanel("Performance Plot", plotlyOutput("modelPerformancePlot")),
                 tabPanel("Summary", verbatimTextOutput("modelPerformanceSummary"))
               )
             )
           )
  ),
  
  # Clustering Results Tab
  tabPanel("Clustering Results",
           sidebarLayout(
             sidebarPanel(
               helpText("k = 3 is the optimal number of clusters based on analysis."),
               # Reintroduce slider for variable k selection
               sliderInput("clusters", "Select Number of Clusters (k):",
                           min = 2, max = 6, value = 3,
                           step = 1,
                           animate = animationOptions(interval = 1000, loop = FALSE))
             ),
             mainPanel(
               tabsetPanel(
                 tabPanel("Cluster Plot", plotlyOutput("clusterPlot")),
                 tabPanel("Cluster Summary", verbatimTextOutput("clusterSummary")),
                 tabPanel("Elbow Method", plotlyOutput("elbowPlot"))
               )
             )
           )
  )
)

# Define Server
server <- function(input, output) {
  
  # ---------------------------
  # Exploratory Data Analysis
  # ---------------------------
  
  # EDA Plot Output
  output$edaPlotUI <- renderUI({
    if(input$eda_plot_type == "Correlation Heatmap"){
      plotlyOutput("edaCorrPlotly", width = "1000px", height = "800px")
    } else {
      plotlyOutput("edaPlot", width = "1000px", height = "600px")
    }
  })
  
  # EDA Plot
  output$edaPlot <- renderPlotly({
    req(input$eda_plot_type)
    
    if(input$eda_plot_type == "Single Variable"){
      req(input$eda_single_var)
      p_hist <- ggplot(data_clean, aes_string(x = input$eda_single_var)) +
        geom_histogram(fill = "steelblue", bins = 30, color = "black", alpha = 0.7) +
        theme_minimal() +
        labs(title = paste("Histogram of", input$eda_single_var),
             x = input$eda_single_var, y = "Frequency") +
        geom_vline(aes_string(xintercept = paste("mean(", input$eda_single_var, ")")), 
                   color = "red", linetype = "dashed")
      
      p_box <- ggplot(data_clean, aes_string(y = input$eda_single_var)) +
        geom_boxplot(fill = "lightblue", alpha = 0.7) +
        theme_minimal() +
        labs(title = paste("Boxplot of", input$eda_single_var),
             y = input$eda_single_var)
      
      subplot(ggplotly(p_hist), ggplotly(p_box), nrows = 2, shareX = TRUE)
      
    } else if(input$eda_plot_type == "Multiple Variables"){
      req(input$eda_x_var, input$eda_y_var)
      p <- ggplot(data_clean, aes_string(x = input$eda_x_var, y = input$eda_y_var)) +
        geom_point(color = "purple", alpha = 0.6) +
        theme_minimal() +
        labs(title = paste(input$eda_x_var, "vs", input$eda_y_var),
             x = input$eda_x_var, y = input$eda_y_var)
      ggplotly(p)
      
    } else if(input$eda_plot_type == "Time Series"){
      req(input$eda_time_var)
      p <- ggplot(data_clean, aes_string(x = "Year", y = input$eda_time_var, color = "Entity")) +
        geom_line(linewidth = 1, alpha = 0.7) +   
        theme_minimal() +
        labs(title = paste(input$eda_time_var, "Over Time"),
             x = "Year",
             y = input$eda_time_var) +
        theme(legend.position = "none")
      ggplotly(p)
    }
  })
  
  # EDA Correlation Heatmap Plotly Using heatmaply
  output$edaCorrPlotly <- renderPlotly({
    req(input$eda_plot_type == "Correlation Heatmap")
    numeric_vars <- data_clean[, sapply(data_clean, is.numeric)]
    
    if(ncol(numeric_vars) < 2){
      plot_ly(type = 'scatter', mode = 'markers') %>%
        layout(title = "Not enough numeric variables for correlation heatmap.")
    } else {
      corr_matrix <- cor(numeric_vars, use = "complete.obs")
      
      heatmaply(
        corr_matrix, 
        colors = viridis::viridis(200), # Enhanced color palette
        main = "Correlation Heatmap",
        xlab = "Variables",
        ylab = "Variables",
        margins = c(100, 100, 40, 20), # Increased margins for better label spacing
        fontsize_row = 12, # Increased row label font size
        fontsize_col = 12, # Increased column label font size
        labRow = rownames(corr_matrix),
        labCol = colnames(corr_matrix),
        dendrogram = "none", # Remove dendrograms for simplicity
        grid_color = "white", # Add grid lines for clarity
        grid_width = 0.5,
        showticklabels = TRUE,
        branch_length = 0,
        key.title = "Correlation",
        key.xlab = "Correlation Coefficient",
        hide_colorbar = FALSE
      )
    }
  })
  
  # EDA Summary
  output$edaSummary <- renderPrint({
    req(input$eda_plot_type)
    
    if(input$eda_plot_type == "Single Variable"){
      req(input$eda_single_var)
      summary(data_clean[[input$eda_single_var]])
      
    } else if(input$eda_plot_type == "Multiple Variables"){
      req(input$eda_x_var, input$eda_y_var)
      cor_val <- cor(data_clean[[input$eda_x_var]], data_clean[[input$eda_y_var]], use = "complete.obs")
      cat("Correlation between", input$eda_x_var, "and", input$eda_y_var, "is:", round(cor_val, 2))
      
    } else if(input$eda_plot_type == "Correlation Heatmap"){
      numeric_vars <- data_clean[, sapply(data_clean, is.numeric)]
      corr_matrix <- cor(numeric_vars, use = "complete.obs")
      print(round(corr_matrix, 2))
      
    } else if(input$eda_plot_type == "Time Series"){
      req(input$eda_time_var)
      summary(data_clean[[input$eda_time_var]])
    }
  })
  
  # ---------------------------
  # Single Variable Model Performance
  # ---------------------------
  
  # Histogram
  output$singleVarHist <- renderPlotly({
    req(input$variable)
    p <- ggplot(model_df, aes_string(x = input$variable)) +
      geom_histogram(fill = "steelblue", bins = 30, color = "black", alpha = 0.7) +
      theme_minimal() +
      labs(title = paste("Histogram of", input$variable),
           x = input$variable, y = "Frequency") +
      geom_vline(aes_string(xintercept = paste("mean(", input$variable, ")")), 
                 color = "red", linetype = "dashed")
    ggplotly(p)
  })
  
  # Boxplot
  output$singleVarBox <- renderPlotly({
    req(input$variable)
    p <- ggplot(model_df, aes_string(y = input$variable)) +
      geom_boxplot(fill = "lightblue", alpha = 0.7) +
      theme_minimal() +
      labs(title = paste("Boxplot of", input$variable),
           y = input$variable)
    ggplotly(p)
  })
  
  # Summary
  output$singleVarSummary <- renderPrint({
    req(input$variable)
    summary(model_df[[input$variable]])
  })
  
  # ---------------------------
  # Classification Models Performance
  # ---------------------------
  
  # Performance Plot
  output$modelPerformancePlot <- renderPlotly({
    req(input$models)
    if(length(input$models) == 0){
      return(NULL)
    }
    
    # Replace with your actual performance data
    # Example placeholder data
    performance_data <- data.frame(
      Model = c("Decision Tree", "Logistic Regression"),
      Accuracy = c(0.85, 0.72),
      Sensitivity = c(0.91, 0.95),
      Specificity = c(0.68, 0.04)
    )
    
    performance_data <- performance_data %>% filter(Model %in% input$models)
    
    # Melt data for plotting
    melted_data <- melt(performance_data, id.vars = "Model")
    
    p <- ggplot(melted_data, aes(x = Model, y = value, fill = variable)) +
      geom_bar(stat = "identity", position = "dodge") +
      theme_minimal() +
      labs(title = "Classification Models Performance",
           x = "Model", y = "Metric Value",
           fill = "Metrics") +
      scale_fill_brewer(palette = "Set1")
    
    ggplotly(p)
  })
  
  # Performance Summary
  output$modelPerformanceSummary <- renderPrint({
    req(input$models)
    if(length(input$models) == 0){
      cat("Please select at least one model to view performance metrics.")
    } else {
      # Replace with your actual performance data
      # Example placeholder data
      performance_data <- data.frame(
        Model = c("Decision Tree", "Logistic Regression"),
        Accuracy = c(0.85, 0.72),
        Sensitivity = c(0.91, 0.95),
        Specificity = c(0.68, 0.04)
      )
      performance_data <- performance_data %>% filter(Model %in% input$models)
      print(performance_data)
    }
  })
  
  # ---------------------------
  # Clustering Results
  # ---------------------------
  
  # Elbow Method Plot
  output$elbowPlot <- renderPlotly({
    # Calculate total within-cluster sum of squares for different k
    wss <- sapply(2:6, function(k){
      kmeans(clustering_data, centers = k, nstart = 25)$tot.withinss
    })
    
    elbow_df <- data.frame(k = 2:6, WSS = wss)
    
    p <- ggplot(elbow_df, aes(x = k, y = WSS)) +
      geom_point(color = "darkgreen", size = 3) +
      geom_line(color = "darkgreen") +
      theme_minimal() +
      labs(title = "Elbow Method for Determining Optimal k",
           x = "Number of Clusters (k)",
           y = "Total Within-Clusters Sum of Squares")
    
    ggplotly(p)
  })
  
  # Cluster Plot
  output$clusterPlot <- renderPlotly({
    req(input$clusters)
    k <- input$clusters
    set.seed(123)
    kmeans_result <- kmeans(clustering_data, centers = k, nstart = 25)
    
    # Perform PCA for visualization
    pca_result <- prcomp(clustering_data, scale. = TRUE) # Ensure data is scaled
    pca_data <- as.data.frame(pca_result$x[,1:2])
    pca_data$Cluster <- factor(kmeans_result$cluster)
    
    # Plot using ggplot2
    p <- ggplot(pca_data, aes(x = PC1, y = PC2, color = Cluster, text = paste("Cluster:", Cluster))) +
      geom_point(alpha = 0.6, size = 2) +
      stat_ellipse(type = "convex") +
      theme_minimal() +
      labs(title = paste("K-means Clustering Results (k =", k, ")"),
           x = "Principal Component 1", y = "Principal Component 2") +
      scale_color_brewer(palette = "Set1")
    
    ggplotly(p, tooltip = "text")
  })
  
  # Cluster Summary
  output$clusterSummary <- renderPrint({
    req(input$clusters)
    k <- input$clusters
    set.seed(123)
    kmeans_result <- kmeans(clustering_data, centers = k, nstart = 25)
    
    # Summarize cluster characteristics
    cluster_centers <- as.data.frame(kmeans_result$centers)
    cluster_sizes <- kmeans_result$size
    
    summary_data <- cbind(cluster_centers, Size = cluster_sizes)
    print(summary_data)
  })
  
}

# Run the application 
shinyApp(ui = ui, server = server)
