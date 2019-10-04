import time

try:
    import numpy
except ImportError:
    raise ImportError('Numpy is not install on your system. Use '
                      'the following command in a terminal: pip install numpy')
try:
    import pygame
except ImportError:
    raise ImportError('pygame is not install on your system. Use '
                      'the following command in a terminal: pip install pygame')
try:
    from matplotlib import pyplot as plt
except ImportError:
    raise ImportError('matplotlib is not install on your system. Use '
                      'the following command in a terminal: pip install matplotlib')
try:
    import cv2
except ImportError:
    raise ImportError('cv2 is not install on your system. Use '
                      'the following command in a terminal: pip install opencv-python')
try:
    import sobel
except ImportError:
    raise ImportError('Sobel library is missing. '
                      'https://github.com/yoyoberenguer/Sobel-Feldman')

"""
A low-pass filter (LPF) is a filter that passes signals 
with a frequency lower than a selected cutoff frequency 
and attenuates signals with frequencies higher than the 
cutoff frequency. 
"""


def lpf():
    image = cv2.imread('seychelles.jpg', 0)
    frequence = numpy.fft.fft2(image)
    freq_shift = numpy.fft.fftshift(frequence)
    rows, cols = image.shape
    rows2, cols2 = rows // 2, cols // 2
    low_freq = freq_shift[rows2 - 30:rows2 + 30, cols2 - 30:cols2 + 30]
    freq_shift = numpy.zeros(image.shape, dtype=numpy.complex_)
    freq_shift[rows2 - 30:rows2 + 30, cols2 - 30:cols2 + 30] = low_freq
    f_ishift = numpy.fft.ifftshift(freq_shift)
    img_back = numpy.fft.ifft2(f_ishift)
    img_back = numpy.abs(img_back)
    plt.subplot(131), plt.imshow(image, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(img_back, cmap='gray')
    plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
    # pygame.image.save(pygame.surfarray.make_surface(
    # img_back.astype(numpy.uint8)), 'seychell_lpf.png')

    plt.show()


"""
A high-pass filter (HPF) is an electronic filter that 
passes signals with a frequency higher than a certain 
cutoff frequency and attenuates signals with frequencies 
lower than the cutoff frequency. The amount of attenuation
for each frequency depends on the filter design. 
A high-pass filter is usually modeled as a linear 
time-invariant system.
"""


def hpf():
    image = cv2.imread('seychelles_gray.jpg', 0)
    print(image.shape)
    frequence = numpy.fft.fft2(image)
    freq_shift = numpy.fft.fftshift(frequence)

    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    # Remove the low frequency from the domain
    freq_shift[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0
    f_ishift = numpy.fft.ifftshift(freq_shift)
    img_back = numpy.fft.ifft2(f_ishift)
    img_back = numpy.abs(img_back)

    plt.subplot(131), plt.imshow(image, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(img_back, cmap='gray')
    plt.title('Image after HPF'), plt.xticks([]), plt.yticks([])
    # pygame.image.save(pygame.surfarray.make_surface(
    # img_back.astype(numpy.uint8)), 'seychell_hps.png')
    plt.show()


if __name__ == '__main__':
    hpf()   # -> Question 4 (remove low frequency)
    lpf()   # -> supplement (remove high frequency)

    """
    Below shows how to apply a Sobel filter.
    See https://github.com/yoyoberenguer/Sobel-Feldman for more image processing (Canny, Prewitt, Sobel) 
    """
    # numpy.set_printoptions(threshold=numpy.nan)

    SIZE = (800, 600)
    SCREENRECT = pygame.Rect((0, 0), SIZE)
    pygame.init()
    SCREEN = pygame.display.set_mode(SCREENRECT.size, pygame.RESIZABLE, 32)
    TEXTURE1 = pygame.image.load("seychelles_gray.jpg").convert()
    TEXTURE1 = pygame.transform.smoothscale(TEXTURE1, (SIZE[0], SIZE[1] >> 1))
    # Texture re-scale to create extra data (padding) on each sides
    PADDING = pygame.transform.smoothscale(TEXTURE1, (SIZE[0] + 8, (SIZE[1] >> 1) + 8))

    Sob = sobel.Sobel4(TEXTURE1, pygame.surfarray.array3d(TEXTURE1))

    FRAME = 0
    clock = pygame.time.Clock()
    STOP_GAME = False
    PAUSE = False

    while not STOP_GAME:

        pygame.event.pump()

        while PAUSE:
            event = pygame.event.wait()
            keys = pygame.key.get_pressed()
            if keys[pygame.K_PAUSE]:
                PAUSE = False
                pygame.event.clear()
                keys = None
            break

        for event in pygame.event.get():

            keys = pygame.key.get_pressed()

            if event.type == pygame.QUIT or keys[pygame.K_ESCAPE]:
                print('Quitting')
                STOP_GAME = True

            elif event.type == pygame.MOUSEMOTION:
                MOUSE_POS = event.pos

            elif keys[pygame.K_PAUSE]:
                PAUSE = True
                print('Paused')

        t = time.time()
        array = Sob.run()
        import hashlib

        # Checking hashes between 2 different method
        hash = hashlib.md5()
        hash.update(array.copy('C'))
        array1 = Sob.run()
        hash_ = hashlib.md5()
        hash_.update(array1.copy('C'))

        print(hash.hexdigest() == hash_.hexdigest())
        print(time.time() - t)

        surface = pygame.surfarray.make_surface(array)

        SCREEN.fill((0, 0, 0, 0))
        SCREEN.blit(TEXTURE1, (0, 0))
        SCREEN.blit(surface, (0, SIZE[1] // 2))

        pygame.display.flip()
        TIME_PASSED_SECONDS = clock.tick(120)
        FRAME += 1

    pygame.quit()
