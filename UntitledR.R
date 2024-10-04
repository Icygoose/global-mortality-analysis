# Read the .Rmd file
lines <- readLines("project.Rmd")

# Extract lines that contain markdown headings (lines that start with #)
headings <- grep("^#+", lines, value = TRUE)

# Display or copy the outline
cat(headings, sep = "\n")
