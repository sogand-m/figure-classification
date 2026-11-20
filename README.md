# Plot analysis in scientific articles

## Articles

In the folder `Articles_ESEM2022` there are the articles from the ESEM 2022 conference.

## Figures

In the folder `Figures_ESEM2022` there are the figures extracted from the articles from the ESEM 2022 conference.

## Script

- find_images: extracts screenshots of figures in a pdf

    Single file:

    ```sh
    python3 find_images.py --output Output/ESEM_2022 --figures Articles_ESEM2022/3544902.3546233.pdf
    ```

    Folder:

    ```sh
    python3 find_images.py --output Output/ESEM_2022 --figures Articles_ESEM2022/
    ```

- describe_image: sample of description extraction using a VLM (through ollama)