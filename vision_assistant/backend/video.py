import cv2, base64
from vision import query_vlm, query_text_only

MAX_FRAMES = 8

def frame_to_b64(frame):
    _, jpg = cv2.imencode(".jpg", frame)
    return "data:image/jpeg;base64," + base64.b64encode(jpg).decode()

def summarize_video(path):
    cap = cv2.VideoCapture(path)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return "Unable to process video."

    step = max(1, total // MAX_FRAMES)
    frame_ids = [i * step for i in range(MAX_FRAMES)]

    changes = []
    prev_scene = None
    first_frame_b64 = None

    for idx, fid in enumerate(frame_ids):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if not ret:
            continue

        b64 = frame_to_b64(frame)

        # Base scene
        if idx == 0:
            first_frame_b64 = b64
            prev_scene = query_vlm(
                b64,
                "Briefly describe the scene. Focus only on visible people, objects, and actions.",
                profile="image_summary"
            )
            changes.append(f"Initial scene: {prev_scene}")
            continue

        # Change-only comparison
        desc = query_vlm(
            b64,
            (
                "You are comparing two video frames.\n"
                f"Previous scene: {prev_scene}\n\n"
                "Rules:\n"
                "- Describe ONLY what changed compared to the previous scene.\n"
                "- Use short phrases.\n"
                "- If nothing changed, reply exactly: no significant change.\n"
            ),
            profile="live"
        )

        if "no significant change" not in desc.lower():
            changes.append(desc)

            # update scene state (append change)
            prev_scene = prev_scene + "; " + desc

    cap.release()

    if len(changes) <= 1:
        return "No significant changes detected throughout the video."

    summary_prompt = (
        "You are given a list of visual changes detected across a video.\n"
        "Create a COMPLETE, chronological summary in full sentences.\n"
        "Do not cut the summary.\n\n"
        "Changes:\n" + "\n".join(f"- {c}" for c in changes)
    )

    # 🔥 TEXT-ONLY FINAL SUMMARY
    return query_text_only(
        "You are summarizing a video.",
        summary_prompt,
        profile="video_summary"
    )
