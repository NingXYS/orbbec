# Security and Robot-Safety Policy

## Credentials

Cloud API keys must be supplied through environment variables or local secret files excluded from Git.

Never commit:

- API keys / access tokens;
- private certificates;
- real `.env` files;
- deployment passwords.

An obsolete credential file was removed from the current branch. As a general rule, any credential ever committed to a public Git repository must be considered exposed and should be revoked or rotated; deleting the latest file does not erase Git history.

## Perception is not a motion command

Camera-frame detections, VLM boxes, 3D centroids and grasp previews are not robot-safe poses by themselves.

Before autonomous execution, validate:

1. camera calibration and RGB-D alignment;
2. hand-eye transform;
3. TCP / tool transform;
4. workspace and reachability;
5. motion-planning and collision constraints;
6. velocity / force limits;
7. emergency-stop readiness;
8. closed-loop acceptance on the real workcell.

The wire-grasp module deliberately reports incomplete states such as missing metric path, ambiguous instance, insufficient wire length or missing support-plane information instead of inventing executable poses.

## Reports

Please report credential exposure, unsafe transform handling, collision / motion-planning bypasses or other safety-relevant defects through a GitHub issue without including secrets or sensitive deployment data.
