from code.segmentor import Segmentor
from code.generator import Generator

def main():
    segmentor = Segmentor()
    segmentor.segment(
        source_images='./DS/images/',
        source_masks='./DS/segments/',
        destination_directory='./output/segmented'
    )

    generator = Generator()
    generator.generate(
        backgrounds='./DS/backgrounds',
        overlays='./output/segmented',
        destination_directory='./output/overlayed_backgrounds'
    )

if __name__ == "__main__":
    main()
