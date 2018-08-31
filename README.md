# ulcer-image-segmentation

Automatic scrapping of ulcer image database and segmentation of the wounded regions.

Databases:

- http://www.medetec.co.uk/slide%20scans/leg-ulcer-images/index.html
- http://www.medetec.co.uk/slide%20scans/leg-ulcer-images-2/index.html
- http://www.medetec.co.uk/slide%20scans/pressure-ulcer-images-a/index.html
- http://www.medetec.co.uk/slide%20scans/pressure-ulcer-images-b/index.html

Running:

1. Execute scrap_images.py

2. Go to [LabelMe](http://labelme.csail.mit.edu/Release3.0/) and annotate images by hand.

3. Save XMLs files in annotations_xml/

4. Execute annotate.py

5. Execute segment.py with the following parameters:

    - `python segment.py --images=data/ --annotations=annotations_images/`
