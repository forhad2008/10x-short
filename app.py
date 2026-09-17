import os
import io
import uuid

from flask import Flask, send_from_directory, request, jsonify
from PIL import Image, ImageOps


# =========================================================
# APP SETUP
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)

# No artificial upload-size limit.
app.config["MAX_CONTENT_LENGTH"] = None

# Target = 10% of original file size.
TARGET_RATIO = 0.10

# The compressor can go lower if necessary to reach the target.
MIN_QUALITY = 30
MAX_QUALITY = 100


# =========================================================
# BASIC HELPERS
# =========================================================

def format_bytes(size):
    if size < 1024:
        return f"{size} B"

    if size < 1024 * 1024:
        return f"{size / 1024:.2f} KB"

    if size < 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024):.2f} MB"

    return f"{size / (1024 * 1024 * 1024):.2f} GB"


def clean_filename(filename):
    filename = os.path.basename(filename)

    name, _ = os.path.splitext(filename)

    name = name.strip()

    if not name:
        name = "image"

    return name[:100]


# =========================================================
# IMAGE PREPARATION
# =========================================================

def prepare_image(image):
    """
    Correct EXIF orientation without cropping.
    """

    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass

    return image


def to_rgb(image):
    """
    Convert an image to RGB for JPEG encoding.

    Transparent images are placed on a white background.
    """

    if image.mode == "RGB":
        return image

    if image.mode in ("RGBA", "LA", "P"):
        rgba = image.convert("RGBA")

        background = Image.new(
            "RGB",
            rgba.size,
            (255, 255, 255)
        )

        background.paste(
            rgba,
            mask=rgba.getchannel("A")
        )

        return background

    return image.convert("RGB")


# =========================================================
# ENCODERS
# =========================================================

def encode_jpeg(image, quality):
    image = to_rgb(image)

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=quality,
        optimize=True,
        progressive=True,
        subsampling=0
    )

    return buffer.getvalue()


def encode_webp(image, quality):
    buffer = io.BytesIO()

    if image.mode not in ("RGB", "RGBA"):
        if "A" in image.mode:
            image = image.convert("RGBA")
        else:
            image = image.convert("RGB")

    image.save(
        buffer,
        format="WEBP",
        quality=quality,
        method=6
    )

    return buffer.getvalue()


# =========================================================
# CREATE FORMAT CANDIDATES
# =========================================================

def create_candidates(image, quality):

    candidates = []

    # -------------------------
    # JPEG
    # -------------------------

    try:
        data = encode_jpeg(
            image,
            quality
        )

        candidates.append({
            "data": data,
            "format": "JPEG",
            "extension": "jpg",
            "size": len(data),
            "quality": quality
        })

    except Exception:
        pass


    # -------------------------
    # WEBP
    # -------------------------

    try:
        data = encode_webp(
            image,
            quality
        )

        candidates.append({
            "data": data,
            "format": "WEBP",
            "extension": "webp",
            "size": len(data),
            "quality": quality
        })

    except Exception:
        pass


    if not candidates:
        raise RuntimeError(
            "Could not encode the image."
        )

    return candidates


def smallest_candidate(image, quality):

    candidates = create_candidates(
        image,
        quality
    )

    return min(
        candidates,
        key=lambda item: item["size"]
    )


# =========================================================
# QUALITY SEARCH
# =========================================================

def find_best_quality(image, target_size):

    best = None

    low = MIN_QUALITY
    high = MAX_QUALITY


    # Test maximum quality first.
    try:
        maximum = smallest_candidate(
            image,
            MAX_QUALITY
        )

        best = maximum

        if maximum["size"] <= target_size:
            return maximum

    except Exception:
        pass


    # Binary search for a good quality.
    for _ in range(9):

        if low > high:
            break

        quality = (low + high) // 2

        try:
            candidate = smallest_candidate(
                image,
                quality
            )
        except Exception:
            break


        if best is None:
            best = candidate
        else:
            current_difference = abs(
                best["size"] - target_size
            )

            new_difference = abs(
                candidate["size"] - target_size
            )

            if new_difference < current_difference:
                best = candidate


        if candidate["size"] > target_size:
            high = quality - 1
        else:
            low = quality + 1


    return best


# =========================================================
# RESIZE WITHOUT CROPPING
# =========================================================

def proportional_resize(image, scale):

    new_width = max(
        1,
        round(image.width * scale)
    )

    new_height = max(
        1,
        round(image.height * scale)
    )


    if (
        new_width == image.width
        and
        new_height == image.height
    ):
        return image


    return image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )


# =========================================================
# SMART 10X COMPRESSOR
# =========================================================

def smart_compress(image, original_size):

    target_size = max(
        1,
        int(original_size * TARGET_RATIO)
    )


    best = None


    # We prefer keeping the original dimensions.
    # Resolution is reduced only when necessary.
    scales = [
        1.00,
        0.97,
        0.94,
        0.91,
        0.88,
        0.85,
        0.82,
        0.79,
        0.76,
        0.73,
        0.70,
        0.67,
        0.64,
        0.61,
        0.58,
        0.55,
        0.52,
        0.49,
        0.46,
        0.43,
        0.40,
        0.37,
        0.34,
        0.31,
        0.28,
        0.25,
        0.22,
        0.19,
        0.16,
        0.13,
        0.10
    ]


    for scale in scales:

        working_image = proportional_resize(
            image,
            scale
        )


        try:
            candidate = find_best_quality(
                working_image,
                target_size
            )
        except Exception:
            continue


        if candidate is None:
            continue


        candidate["width"] = working_image.width
        candidate["height"] = working_image.height
        candidate["scale"] = scale


        if best is None:

            best = candidate

        else:

            old_difference = abs(
                best["size"] - target_size
            )

            new_difference = abs(
                candidate["size"] - target_size
            )

            if new_difference < old_difference:
                best = candidate


        # Once we have reached the target at a
        # reasonably large resolution, stop.
        if (
            candidate["size"] <= target_size
            and
            scale >= 0.40
        ):
            break


    if best is None:
        raise RuntimeError(
            "Unable to compress this image."
        )


    return best, target_size


# =========================================================
# SAVE OUTPUT
# =========================================================

def save_output(candidate, original_filename):

    base_name = clean_filename(
        original_filename
    )

    unique_id = uuid.uuid4().hex[:10]

    filename = (
        base_name
        + "_10x_"
        + unique_id
        + "."
        + candidate["extension"]
    )

    path = os.path.join(
        OUTPUT_FOLDER,
        filename
    )


    with open(path, "wb") as file:
        file.write(candidate["data"])


    return filename


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    return send_from_directory(
        BASE_DIR,
        "index.html"
    )


# =========================================================
# COMPRESS API
# =========================================================

@app.route(
    "/resize",
    methods=["POST"]
)
def resize():

    try:

        if "image" not in request.files:

            return jsonify({
                "success": False,
                "error": "No image was uploaded."
            }), 400


        uploaded = request.files["image"]


        if not uploaded.filename:

            return jsonify({
                "success": False,
                "error": "Please select an image."
            }), 400


        # -------------------------------------------------
        # Read uploaded image
        # -------------------------------------------------

        original_data = uploaded.read()


        if not original_data:

            return jsonify({
                "success": False,
                "error": "The uploaded file is empty."
            }), 400


        original_size = len(
            original_data
        )


        # -------------------------------------------------
        # Open image
        # -------------------------------------------------

        image_stream = io.BytesIO(
            original_data
        )

        image = Image.open(
            image_stream
        )

        image.load()


        # Correct orientation.
        image = prepare_image(
            image
        )


        original_width = image.width
        original_height = image.height

        original_format = (
            image.format
            or
            "Unknown"
        )


        # -------------------------------------------------
        # COMPRESS
        # -------------------------------------------------

        candidate, target_size = smart_compress(
            image,
            original_size
        )


        # -------------------------------------------------
        # SAVE
        # -------------------------------------------------

        output_filename = save_output(
            candidate,
            uploaded.filename
        )


        output_path = os.path.join(
            OUTPUT_FOLDER,
            output_filename
        )


        new_size = os.path.getsize(
            output_path
        )


        # -------------------------------------------------
        # STATISTICS
        # -------------------------------------------------

        reduction_percent = (
            1 -
            (
                new_size /
                original_size
            )
        ) * 100


        remaining_percent = (
            new_size /
            original_size
        ) * 100


        target_reached = (
            new_size <= target_size
        )


        return jsonify({

            "success": True,

            "filename": output_filename,

            "preview_url":
                "/outputs/"
                + output_filename,

            "download_url":
                "/download/"
                + output_filename,

            "original_size":
                original_size,

            "new_size":
                new_size,

            "target_size":
                target_size,

            "reduction_percent":
                reduction_percent,

            "remaining_percent":
                remaining_percent,

            "target_reached":
                target_reached,

            "original_width":
                original_width,

            "original_height":
                original_height,

            "width":
                candidate["width"],

            "height":
                candidate["height"],

            "format":
                candidate["format"],

            "quality":
                candidate["quality"],

            "scale":
                candidate["scale"],

            "original_format":
                original_format

        })


    except Exception as error:

        print(
            "IMAGE ERROR:",
            repr(error)
        )


        return jsonify({

            "success": False,

            "error":
                "Could not process the image: "
                +
                str(error)

        }), 500


# =========================================================
# PREVIEW
# =========================================================

@app.route(
    "/outputs/<path:filename>"
)
def preview(filename):

    return send_from_directory(
        OUTPUT_FOLDER,
        filename
    )


# =========================================================
# DOWNLOAD
# =========================================================

@app.route(
    "/download/<path:filename>"
)
def download(filename):

    return send_from_directory(
        OUTPUT_FOLDER,
        filename,
        as_attachment=True
    )


# =========================================================
# START SERVER
# =========================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )