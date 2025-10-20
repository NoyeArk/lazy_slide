from wsidata import open_wsi

slide = "data/GTEX-1117F-0526.svs"

# Now we'll open the slide using the open_wsi function from wsidata
# This creates a WSIData object that contains both the image and associated metadata
wsi = open_wsi(slide)

# Let's examine what's in our WSI object
print(wsi)
