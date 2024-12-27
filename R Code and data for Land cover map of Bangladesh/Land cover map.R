# Set the working directory
setwd("C:/Users/16808/Desktop/A. Weather dataset/Land cover map")

# Verify the working directory has been set correctly
getwd()

# Load the necessary packages
libs <- c(
  "terra",
  "giscoR",
  "sf",
  "tidyverse",
  "ggtern",
  "elevatr",
  "png",
  "rayshader",
  "magick",
  "rgl"
)

installed_libraries <- libs %in% rownames(installed.packages())

if(any(installed_libraries == FALSE)){
  install.packages(libs[!installed_libraries])
}

invisible(lapply(libs, library, character.only = TRUE))

# Disable RGL device
options(rgl.useNULL = TRUE)

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

# DIGITAL ELEVATION MODEL
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

# RENDER SCENE
elev_lambert <- elev |>
  terra::rast() |>
  terra::project(crs_lambert)

elmat <- rayshader::raster_to_matrix(
  elev_lambert
)

h <- nrow(elev_lambert)
w <- ncol(elev_lambert)

elmat |>
  rayshader::height_shade(
    texture = colorRampPalette(
      cols[9]
    )(256)
  ) |>
  rayshader::add_overlay(
    img,
    alphalayer = 1
  ) |>
  rayshader::plot_3d(
    elmat,
    zscale = 12,
    solid = F,
    shadow = T,
    shadow_darkness = 1,
    background = "white",
    windowsize = c(
      w / 5, h / 5
    ),
    zoom = 0.67,
    phi = 85,
    theta = 0
  )

rayshader::render_camera(
  zoom = .58
)

# RENDER OBJECT
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
  parallel = T,
  width = w * 1.5,
  height = h * 1.5
)

# PUT EVERYTHING TOGETHER
legend_name <- "land_cover_legend.png"
png(legend_name)
par(family = "mono")

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
    "Water",
    "Trees",
    "Crops",
    "Built area",
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
  gravity = "southwest",
  offset = "+100+0"
)

magick::image_write(
  p, "3d_bangladesh_land_cover_final.png"
)

