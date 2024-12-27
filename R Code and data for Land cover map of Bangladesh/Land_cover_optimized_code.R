# Set the working directory
setwd("C:/Users/16808/Desktop/A. Weather dataset/Land cover map")

# Verify the working directory has been set correctly
getwd()

# 1. Packages

libs <- c(
  "terra",
  "giscoR",
  "sf",
  "tidyverse",
  "ggtern",
  "elevatr",
  "png",
  "rayshader",
  "magick"
)

installed_libraries <- libs %in% rownames(installed.packages())

if(any(installed_libraries == FALSE)){
  install.packages(libs[!installed_libraries])
}

invisible(lapply(libs, library, character.only = TRUE))

# Country Border
country_sf <- giscoR::gisco_get_countries(
  country = "BD",
  resolution = "1"
)

# List all .tif files in the current directory
raster_files <- list.files(
  path = getwd(),
  pattern = "tif",
  full.names = TRUE
)

# Define the CRS
crs <- "EPSG:4326"

for (raster in raster_files) {
  # Read the raster file
  rasters <- terra::rast(raster)
  
  # Transform the country shapefile to the raster's CRS
  country <- country_sf |>
    sf::st_transform(crs = terra::crs(rasters))
  
  # Crop and mask the raster with the country's boundary
  land_cover <- rasters |>
    terra::crop(vect(country), snap = "in", mask = TRUE) |>
    terra::aggregate(fact = 5, fun = "modal")
  
  # Project the raster to the specified CRS
  land_cover <- terra::project(land_cover, crs)
  
  # Write the processed raster to a new file
  terra::writeRaster(
    land_cover,
    paste0(tools::file_path_sans_ext(raster), "_bangladesh", ".tif"),
    overwrite = TRUE
  )
}

# Load virtual layer
r_list <- list.files(
  path = getwd(),
  pattern = "_bangladesh.tif",
  full.names = TRUE
)

land_cover_vrt <- terra::vrt(
  r_list,
  "bangladesh_land_cover_vrt.vrt",
  overwrite = TRUE
)

# Fetch Original colors
ras <- terra::rast(raster_files[[1]])

raster_color_table <- do.call(
  data.frame,
  terra::coltab(ras)[[1]]
)

# Convert RGB to Hex
hex_code <- ggtern::rgb2hex(
  r = raster_color_table[,2],
  g = raster_color_table[,3],
  b = raster_color_table[,4]
)

print(raster_color_table)
print(hex_code)

# ASSIGN COLORS TO RASTER

c("#000000", "#419bdf", "#397d49",
  "#000000", "#7a87c6", "#e49635",
  "#000000", "#c4281b", "#a59b8f",
  "#a8ebff", "#616161", "#e3e2c3")

cols <- hex_code[c(2:3, 5:6, 8:12)]

from <- c(1:2, 4:5, 7:11)
to <- t(col2rgb(cols))
land_cover_vrt <- na.omit(land_cover_vrt)

land_cover_bangladesh <- terra::subst(
  land_cover_vrt, 
  from = from,
  to = to,
  names = cols
)

terra::plotRGB(land_cover_bangladesh)


# 8. DIGITAL ELEVATION MODEL

elev <- elevatr::get_elev_raster(
  locations = country_sf,
  z = 9, clip = "locations"
)

crs_lambert <- "+proj=laea +lat_0=23.6850 +lon_0=90.3563 +x_0=4321000 +y_0=3210000 +datum=WGS84 +units=m +no_defs"

land_cover_bangladesh_resampled <- terra::resample(
  x = land_cover_bangladesh,
  y = terra::rast(elev),
  method = "near"
) |>
  terra::project(crs_lambert)

terra::plotRGB(land_cover_bangladesh_resampled)


img_file <- "land_cover_bangladesh.png"

terra::writeRaster(
  land_cover_bangladesh_resampled,
  img_file,
  overwrite = T,
  NAflag = 255
)

img <- png::readPNG(img_file)

# 9. RENDER SCENE
#----------------

elev_lambert <- elev |>
  terra::rast() |>
  terra::project(crs_lambert)

elmat <- rayshader::raster_to_matrix(
  elev_lambert
)

h <- nrow(elev_lambert)
w <- ncol(elev_lambert)

# Reduce the resolution of the elmat matrix (if possible)
elmat <- elmat[seq(1, nrow(elmat), by = 2), seq(1, ncol(elmat), by = 2)]

# Simplify the texture
simpler_texture <- colorRampPalette(cols[9])(128)

# Reduce the windowsize parameter
new_windowsize <- c(w / 10, h / 10)

elmat |>
  rayshader::height_shade(
    texture = simpler_texture
  ) |>
  rayshader::add_overlay(
    img,
    alphalayer = 1,  # Adjust the alpha layer if necessary
    rescale_original = TRUE  # Ensure the overlay is rescaled to match elmat
  ) |>
  rayshader::plot_3d(
    elmat,
    zscale = 12,
    solid = F,
    shadow = T,
    shadow_darkness = 1,
    background = "white",
    windowsize = new_windowsize,
    zoom = 0.5,   # Adjust zoom if necessary
    phi = 85,      # Adjust phi if necessary
    theta = 0      # Adjust theta if necessary
  )

rayshader::render_camera(
  zoom = .58  # Ensure this matches the plot_3d parameters if necessary
)

# 10. RENDER OBJECT
#-----------------

hdri_file <- "C:/Users/16808/Desktop/A. Weather dataset/Land cover map/air_museum_playground_4k.hdr"

filename <- "3d_land_cover_bangladesh-dark.png"

rayshader::render_highquality(
  filename = filename,
  preview = T,
  light = F,
  environment_light = hdri_file,
  intensity_env = 1,
  rotate_env = 90,
  interactive = F,
  parallel = F,
  width = w * 1.5,
  height = h * 1.5
)


# 11. PUT EVERYTHING TOGETHER

c(
  "#419bdf", "#397d49", "#7a87c6", 
  "#e49635", "#c4281b", "#a59b8f", 
  "#a8ebff", "#616161", "#e3e2c3"
)


legend_name <- "land_cover_legend.png"
png(legend_name, bg = "transparent")
par(family = "mono", font = 2)

plot(
  NULL,
  xaxt = "n",
  yaxt = "n",
  bty = "n",
  ylab = "",
  xlab = "",
  xlim = 0:1,
  ylim = 0:1,
  xaxs = "i",
  yaxs = "i"
)
legend(
  "center",
  legend = c(
    "Water bodies",
    "Vegatation",
    "Crops",
    "Built-up area",
    "Rangeland"
  ),
  pch = 15,
  cex = 2,
  pt.cex = 1,
  bty = "n",
  col = c(cols[c(1:2, 4:5, 9)]),
  fill = c(cols[c(1:2, 4:5, 9)]),
  border = "grey20"
)
dev.off()

# filename <- "land-cover-bih-3d-b.png"

lc_img <- magick::image_read(
  filename
)

my_legend <- magick::image_read(
  legend_name
)

my_legend_scaled <- magick::image_scale(
  magick::image_background(
    my_legend, "none"
  ), 2500
)

p <- magick::image_composite(
  magick::image_scale(
    lc_img, "x7000" 
  ),
  my_legend_scaled,
  gravity = "northeast",
  offset = "+100+0"
)

magick::image_write(
  p, "3d_bangladesh_land_cover_final.png"
)
