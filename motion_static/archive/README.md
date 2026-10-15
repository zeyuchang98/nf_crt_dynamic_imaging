# Removed legacy implementation

The duplicate `before_reorganization/` package and its generated outputs were
removed during the approved cleanup. The subsequent motion-static history reset
also removes the supported Git recovery route for its 27 legacy Python files.

Use the maintained [implementation map](../README.md) and `python -m motion_static.run`.
No active module imports the removed legacy package. The old directly supervised
motion fit was a historical experiment, not the current nested representation.
