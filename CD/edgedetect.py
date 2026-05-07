import cv2
from skimage import io, filters, color
import matplotlib.pyplot as plt

def main():
    # Nama file gambar (pastikan file ada di folder yang sama dengan script ini)
    image_path = 'UTS/CD/src/eiffel.webp'

    # --- 1. PROSES OPENCV (CANNY) ---
    img_cv2 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_cv2 is None:
        print("Error: Gambar tidak ditemukan!")
        return
    
    edges_canny = cv2.Canny(img_cv2, 100, 200)
    plt.imsave('UTS/CD/src/canny-output.png', edges_canny)

    # --- 2. PROSES SOBEL MENGGUNAKAN SKIMAGE ---
    img_skimage = io.imread(image_path)
    gray_img = color.rgb2gray(img_skimage)
    edges_sobel = filters.sobel(gray_img)
    plt.imsave('UTS/CD/src/sobel-output.png', edges_sobel)

    # --- 3. MENAMPILKAN HASIL KOMPARASI ---
    # Membuat figure dengan 1 baris, 3 kolom
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    # Tampilkan Gambar Asli
    img_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    axes[0].imshow(img_rgb)
    axes[0].set_title('Gambar Asli', fontsize=14)
    axes[0].axis('off')

    # Tampilkan Hasil OpenCV (Canny)
    axes[1].imshow(edges_canny, cmap='gray')
    axes[1].set_title('OpenCV: Canny Edge', fontsize=14)
    axes[1].axis('off')

    # Tampilkan Hasil scikit-image (Sobel)
    axes[2].imshow(edges_sobel, cmap='gray')
    axes[2].set_title('skimage: Sobel Filter', fontsize=14)
    axes[2].axis('off')

    # Rapikan layout dan tampilkan
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()