import os
import uuid
import io

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    send_from_directory
)

from PIL import Image, ImageOps


app = Flask(__name__)

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

OUTPUT_FOLDER = os.path.join(
    BASE_DIR,
    "outputs"
)

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# =========================================================
# MAIN SETTINGS
# =========================================================

TARGET_RATIO = 0.10
# 10% of original = 10x smaller

MIN_QUALITY = 35
MAX_QUALITY = 100

# No hard upload-size restriction.
app.config["MAX_CONTENT_LENGTH"] = None


# =========================================================
# FORMAT HELPERS
# =========================================================

def extension_from_name(filename):

    if "." in filename:
        return filename.rsplit(
            ".",
            1
        )[1].lower()

    return "jpg"


def format_bytes(size):

    if size < 1024:
        return f"{size} B"

    if size < 1024 * 1024:
        return f"{size / 1024:.2f} KB"

    if size < 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024):.2f} MB"

    return f"{size / (1024 * 1024 * 1024):.2f} GB"


def clean_name(filename):

    base = os.path.basename(filename)

    name, _ = os.path.splitext(base)

    return name[:80]


# =========================================================
# IMAGE PREPARATION
# =========================================================

def prepare_rgb(image):

    """
    Convert image to RGB for JPEG encoding.
    Transparency is placed over white.
    """

    if image.mode == "RGB":
        return image

    if image.mode in ("RGBA", "LA"):

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

    if image.mode == "P":

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


def prepare_image(image):

    """
    Fix EXIF orientation without cropping.
    """

    try:

        image = ImageOps.exif_transpose(
            image
        )

    except Exception:
        pass

    return image


# =========================================================
# ENCODING
# =========================================================

def encode_jpeg(image, quality):

    image = prepare_rgb(image)

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

    if image.mode not in (
        "RGB",
        "RGBA"
    ):

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
# CREATE CANDIDATE
# =========================================================

def make_candidates(image, quality):

    candidates = []

    # JPEG
    try:

        jpeg_data = encode_jpeg(
            image,
            quality
        )

        candidates.append({
            "data": jpeg_data,
            "format": "JPEG",
            "extension": "jpg",
            "size": len(jpeg_data)
        })

    except Exception:
        pass


    # WEBP
    try:

        webp_data = encode_webp(
            image,
            quality
        )

        candidates.append({
            "data": webp_data,
            "format": "WEBP",
            "extension": "webp",
            "size": len(webp_data)
        })

    except Exception:
        pass


    if not candidates:

        raise RuntimeError(
            "The image could not be encoded."
        )


    return candidates


# =========================================================
# BEST ENCODING
# =========================================================

def best_encoding(image, quality):

    candidates = make_candidates(
        image,
        quality
    )

    return min(
        candidates,
        key=lambda x: x["size"]
    )


# =========================================================
# SMART QUALITY SEARCH
# =========================================================

def find_quality(image, target_bytes):

    """
    Search for the highest possible quality that
    gets close to the target size.

    No fixed 92% quality requirement.
    """

    low = MIN_QUALITY
    high = MAX_QUALITY

    best = None

    # First test quality 100
    try:

        result = best_encoding(
            image,
            100
        )

        best = result

        if result["size"] <= target_bytes:

            return result

    except Exception:
        pass


    # Binary quality search
    for _ in range(8):

        if low > high:
            break

        middle = (
            low + high
        ) // 2


        try:

            candidate = best_encoding(
                image,
                middle
            )

        except Exception:

            break


        if (
            best is None
            or
            abs(
                candidate["size"]
                -
                target_bytes
            )
            <
            abs(
                best["size"]
                -
                target_bytes
            )
        ):

            best = candidate


        if candidate["size"] > target_bytes:

            high = middle - 1

        else:

            low = middle + 1


    return best


# =========================================================
# SMART RESOLUTION SEARCH
# =========================================================

def resize_proportionally(
    image,
    scale
):

    width = max(
        1,
        round(
            image.width * scale
        )
    )

    height = max(
        1,
        round(
            image.height * scale
        )
    )


    if (
        width == image.width
        and
        height == image.height
    ):

        return image


    return image.resize(
        (
            width,
            height
        ),
        Image.Resampling.LANCZOS
    )


def smart_compress(
    image,
    target_bytes
):

    """
    Main compression engine.

    Strategy:

    1. Try original dimensions.
    2. Find highest quality near target.
    3. If still too large, gradually reduce resolution.
    4. Re-check quality at every resolution.
    5. Keep the candidate closest to target.
    """

    best_overall = None


    # -----------------------------------------------------
    # Resolution scales
    # -----------------------------------------------------

    scales = [

        1.00,

        0.95,
        0.90,
        0.85,
        0.80,
        0.75,
        0.70,
        0.65,
        0.60,
        0.55,
        0.50,

        0.45,
        0.40,
        0.35,
        0.30,
        0.25,
        0.20,
        0.15,
        0.10

    ]


    for scale in scales:

        working_image = resize_proportionally(
            image,
            scale
        )


        try:

            candidate = find_quality(
                working_image,
                target_bytes
            )

        except Exception:

            continue


        if candidate is None:
            continue


        candidate["width"] = (
            working_image.width
        )

        candidate["height"] = (
            working_image.height
        )

        candidate["scale"] = scale


        # -------------------------------------------------
        # First valid candidate
        # -------------------------------------------------

        if best_overall is None:

            best_overall = candidate

        else:

            current_difference = abs(
                best_overall["size"]
                -
                target_bytes
            )

            new_difference = abs(
                candidate["size"]
                -
                target_bytes
            )


            if new_difference < current_difference:

                best_overall = candidate


        # -------------------------------------------------
        # If target reached, stop.
        #
        # Because higher resolution is preferred,
        # this gives us the best quality/resolution
        # result found so far.
        # -------------------------------------------------

        if candidate["size"] <= target_bytes:

            # Don't immediately stop at a tiny file.
            # Continue one more nearby scale if useful.

            if scale >= 0.50:

                break


    if best_overall is None:

        raise RuntimeError(
            "Unable to compress this image."
        )


    return best_overall


# =========================================================
# SAVE OUTPUT
# =========================================================

def save_result(
    candidate,
    filename
):

    output_name = (
        filename
        +
        "-"
        +
        uuid.uuid4().hex[:8]
        +
        "."
        +
        candidate["extension"]
    )


    output_path = os.path.join(
        OUTPUT_FOLDER,
        output_name
    )


    with open(
        output_path,
        "wb"
    ) as file:

        file.write(
            candidate["data"]
        )


    return (
        output_name,
        output_path
    )


# =========================================================
# HOME
# =========================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# =========================================================
# RESIZE
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
                "error": "No image selected."
            }), 400


        uploaded = request.files["image"]


        if not uploaded.filename:

            return jsonify({
                "success": False,
                "error": "Please select an image."
            }), 400


        # -------------------------------------------------
        # Read directly into memory
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

        source = io.BytesIO(
            original_data
        )


        image = Image.open(
            source
        )


        image.load()


        image = prepare_image(
            image
        )


        original_width = image.width
        original_height = image.height


        # -------------------------------------------------
        # TARGET = 10%
        # -------------------------------------------------

        target_bytes = max(
            1,
            int(
                original_size *
                TARGET_RATIO
            )
        )


        # -------------------------------------------------
        # COMPRESS
        # -------------------------------------------------

        result = smart_compress(
            image,
            target_bytes
        )


        # -------------------------------------------------
        # SAVE
        # -------------------------------------------------

        clean_filename = clean_name(
            uploaded.filename
        )


        output_name, output_path = save_result(
            result,
            clean_filename
        )


        final_size = os.path.getsize(
            output_path
        )


        # -------------------------------------------------
        # STATS
        # -------------------------------------------------

        actual_percent = (
            final_size /
            original_size
        ) * 100


        reduction_percent = (
            1 -
            (
                final_size /
                original_size
            )
        ) * 100


        # -------------------------------------------------
        # TARGET DISTANCE
        # -------------------------------------------------

        target_reached = (
            final_size <= target_bytes
        )


        return jsonify({

            "success": True,

            "filename":
                output_name,

            "preview_url":
                "/outputs/"
                +
                output_name,

            "download_url":
                "/download/"
                +
                output_name,

            "original_size":
                original_size,

            "new_size":
                final_size,

            "target_size":
                target_bytes,

            "reduction_percent":
                reduction_percent,

            "actual_percent":
                actual_percent,

            "original_width":
                original_width,

            "original_height":
                original_height,

            "width":
                result["width"],

            "height":
                result["height"],

            "quality":
                result.get(
                    "quality",
                    None
                ),

            "format":
                result["format"],

            "scale":
                result["scale"],

            "target_reached":
                target_reached,

            "original_format":
                image.format or "Unknown"

        })


    except Exception as error:

        print(
            "Compression error:",
            repr(error)
        )


        return jsonify({

            "success": False,

            "error":
                "Could not process this image: "
                +
                str(error)

        }), 500


# =========================================================
# PREVIEW
# =========================================================

@app.route(
    "/outputs/<path:filename>"
)
def output_file(filename):

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
# RUN
# =========================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )