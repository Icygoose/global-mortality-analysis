## **1.4. Shiny App**

To provide an interactive data exploration experience, a Shiny app was developed to dynamically visualize the health metrics across different countries and years. The app allows users to explore single-variable, multi-variable, time-series plots, and a correlation heatmap, providing a comprehensive view of the dataset.

**App Overview**
  
  The Shiny app includes the following features:
  
  - **Single Variable Visualisation**: Users can select a single health metric, such as *Outdoor Air Pollution* or *High Systolic Blood Pressure*, and view its trend over the selected time period for a specific country. This feature enables users to observe how a particular health factor changes over time in the context of a specific country.
- **Multiple Variable Visualisation**: The app also allows users to explore relationships between two health metrics. For example, users can compare *Smoking Rates* against *High Systolic Blood Pressure* or *Diet High in Sodium* against *Outdoor Air Pollution*. This feature provides an intuitive way to observe potential correlations between various health metrics.
- **Time Series Visualisation**: This feature provides an overall view of a selected health metric across all countries over time. It helps visualize how a health issue, like *Smoking Rates* or *High Body Mass Index*, evolves globally, offering insights into trends across different regions.
- **Correlation Heatmap**: To further analyze relationships between multiple health metrics, the app includes a correlation heatmap. The heatmap visually represents the degree of correlation between each pair of health metrics in the dataset, with darker colors representing stronger correlations (positive or negative). This provides a clear overview of how various factors relate to each other across the dataset.

**User Experience**
  
  Users can select the variables they wish to visualize, filter by country and year, and interact with the data dynamically. The app provides flexibility by allowing users to switch between different visualizations and choose health metrics of interest from dropdown menus.

```{r shiny-data, cache=TRUE}
# Load required libraries
library(shiny)
library(reshape2)

# Get all health-related columns
health_metrics <- names(data_clean)[!names(data_clean) %in% c("Entity", "Code", "Year")]

# UI for the Shiny App
ui <- fluidPage(
  titlePanel("Interactive Health Data Visualization"),
  sidebarLayout(
    sidebarPanel(
      selectInput("variable1", "Select First Variable:", 
                  choices = health_metrics),
      selectInput("variable2", "Select Second Variable (for Multi-Variable Plot):", 
                  choices = health_metrics, selected = "High.systolic.blood.pressure"),
      selectInput("country", "Select a Country:", 
                  choices = unique(data_clean$Entity)),
      sliderInput("year", "Select Year Range:",
                  min = min(data_clean$Year), 
                  max = max(data_clean$Year),
                  value = c(min(data_clean$Year), max(data_clean$Year)),
                  step = 1)
    ),
    mainPanel(
      tabsetPanel(
        tabPanel("Single Variable Plot", plotOutput("singlePlot")),
        tabPanel("Multiple Variable Plot", plotOutput("multiPlot")),
        tabPanel("Time Series Plot", plotOutput("timeSeriesPlot")),
        tabPanel("Correlation Heatmap", plotOutput("heatmapPlot"))
      )
    )
  )
)

# Server logic for the Shiny App
server <- function(input, output) {
  # Single Variable Plot (Dynamic)
  output$singlePlot <- renderPlot({
    filtered_data <- data_clean %>%
      filter(Entity == input$country & Year >= input$year[1] & Year <= input$year[2])
    
    ggplot(filtered_data, aes_string(x = "Year", y = input$variable1)) +
      geom_line(color = "blue") +
      labs(title = paste(input$variable1, "Over Time for", input$country),
           x = "Year", y = input$variable1)
  })
  
  # Multiple Variable Plot (Dynamic)
  output$multiPlot <- renderPlot({
    filtered_data <- data_clean %>%
      filter(Entity == input$country & Year >= input$year[1] & Year <= input$year[2])
    
    ggplot(filtered_data, aes_string(x = input$variable1, y = input$variable2)) +
      geom_point(color = "purple") +
      labs(title = paste(input$variable1, "vs", input$variable2, "for", input$country),
           x = input$variable1, y = input$variable2)
  })
  
  # Time Series Plot for selected variable
  output$timeSeriesPlot <- renderPlot({
    filtered_data <- data_clean %>%
      filter(Year >= input$year[1] & Year <= input$year[2])
    
    ggplot(filtered_data, aes(x = Year, y = !!as.symbol(input$variable1), color = Entity)) +
      geom_line() +
      labs(title = paste(input$variable1, "Over Time Across Countries"),
           x = "Year", y = input$variable1)
  })
  
  # Correlation Heatmap
  output$heatmapPlot <- renderPlot({
    correlation_matrix <- cor(data_clean %>%
                                select(-Entity, -Code, -Year), use = "complete.obs")
    
    ggplot(melt(correlation_matrix), aes(Var1, Var2, fill = value)) +
      geom_tile() +
      scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
      labs(title = "Correlation Heatmap", x = "", y = "")
  })
}

# Run the Shiny App
shinyApp(ui = ui, server = server)
```