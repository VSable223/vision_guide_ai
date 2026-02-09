last_scene = ""

def is_new_scene(desc):
    global last_scene
    if desc == last_scene:
        return False
    last_scene = desc
    return True
