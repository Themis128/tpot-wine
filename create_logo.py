from PIL import Image, ImageDraw, ImageFont, ImageOps
from pathlib import Path

def create_logo():
    output_dir = Path("assets")
    output_dir.mkdir(parents=True, exist_ok=True)

    width, height = 500, 150
    bg_color = (255, 255, 255)
    text_color = (80, 20, 20)

    img = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    try:
        font_main = ImageFont.truetype("DejaVuSans-Bold.ttf", 38)
        font_sub = ImageFont.truetype("DejaVuSans.ttf", 16)
    except:
        font_main = ImageFont.load_default()
        font_sub = ImageFont.load_default()

    brand = "cloudless"
    subtitle = "Wine Forecasting Platform"
    signature = "Baltzakis Themistoklis"

    draw.text((20, 30), brand, font=font_main, fill=text_color)
    draw.text((22, 78), subtitle, font=font_sub, fill=(100, 100, 100))
    draw.text((22, 105), signature, font=font_sub, fill=(80, 80, 80))

    # Save PNG
    png_path = output_dir / "logo.png"
    img.save(png_path, "PNG")

    # Save JPEG
    jpg_path = output_dir / "logo.jpg"
    img.convert("RGB").save(jpg_path, "JPEG", quality=90)

    # Save grayscale version (optional use)
    bw_path = output_dir / "logo_bw.png"
    gray_img = ImageOps.grayscale(img)
    gray_img.save(bw_path)

    # Save PDF
    pdf_path = output_dir / "logo.pdf"
    img.save(pdf_path, "PDF", resolution=100.0)

    print("✅ Logos saved:")
    print(f"  - PNG: {png_path}")
    print(f"  - JPG: {jpg_path}")
    print(f"  - BW PNG: {bw_path}")
    print(f"  - PDF: {pdf_path}")

if __name__ == "__main__":
    create_logo()
