import pathlib
import pprint
from copy import deepcopy

# import dex_teleop_parameters as dt
import numpy as np
from urdf_parser_py import urdf as ud


def save_urdf_file(robot, file_name):
    urdf_string = robot.to_xml_string()
    print("Saving new URDF file to", file_name)
    fid = open(file_name, "w")
    fid.write(urdf_string)
    fid.close()
    print("Finished saving")


urdf_filename = (
    "/home/ubuntu/partnr-planner/data/robots/hab_stretch/urdf/hab_stretch.urdf"
)

non_fixed_joints = [
    "joint_lift",
    "joint_arm_l0",
    "joint_arm_l1",
    "joint_arm_l2",
    "joint_arm_l3",
    "joint_wrist_yaw",
    "joint_wrist_pitch",
    "joint_wrist_roll",
]

ik_joint_limits = {
    "joint_mobile_base_translation": (-1, 1),
    # "joint_mobile_base_rotation": (-(np.pi / 2.0), np.pi / 2.0),
    "joint_lift": (0.0, 1.1),
    "joint_arm_l0": (0.0, 0.13),
    "joint_arm_l1": (0.0, 0.13),
    "joint_arm_l2": (0.0, 0.13),
    "joint_arm_l3": (0.0, 0.13),
    "joint_wrist_yaw": (-1.75, 4),
    "joint_wrist_pitch": (-1.57, 0.56),
    "joint_wrist_roll": (-3.14, 3.14),
}

print()
print("Loading URDF from:")
print(urdf_filename)
print("The specialized URDFs will be derived from this URDF.")
robot = ud.Robot.from_xml_file(urdf_filename)

# Change any joint that should be immobile for end effector IK into a fixed joint
for j in robot.joint_map.keys():
    if j not in non_fixed_joints:
        joint = robot.joint_map[j]
        # print('(joint name, joint type) =', (joint.name, joint.type))
        joint.type = "fixed"

robot_rotary = robot
robot_prismatic = deepcopy(robot)

###############################################
# ADD VIRTUAL ROTARY JOINT FOR MOBILE BASE

# Add a virtual base link
link_virtual_base_rotary = ud.Link(
    name="virtual_base", visual=None, inertial=None, collision=None, origin=None
)

# Add rotary joint for the mobile base
origin_rotary = ud.Pose(xyz=[0, 0, 0], rpy=[0, 0, 0])

limit_rotary = ud.JointLimit(effort=10, velocity=1, lower=-np.pi, upper=np.pi)

joint_mobile_base_rotation = ud.Joint(
    name="joint_mobile_base_rotation",
    parent="virtual_base",
    child="base_link",
    joint_type="revolute",
    axis=[0, 0, 1],
    origin=origin_rotary,
    limit=limit_rotary,
    dynamics=None,
    safety_controller=None,
    calibration=None,
    mimic=None,
)

robot_rotary.add_link(link_virtual_base_rotary)
robot_rotary.add_joint(joint_mobile_base_rotation)
###############################################


###############################################
# ADD VIRTUAL PRISMATIC JOINT FOR MOBILE BASE

# Add a virtual base link
link_virtual_base_prismatic = ud.Link(
    name="virtual_base", visual=None, inertial=None, collision=None, origin=None
)

# Add rotary joint for the mobile base
origin_prismatic = ud.Pose(xyz=[0, 0, 0], rpy=[0, 0, 0])

limit_prismatic = ud.JointLimit(effort=10, velocity=1, lower=-1.0, upper=1.0)

joint_mobile_base_translation = ud.Joint(
    name="joint_mobile_base_translation",
    parent="virtual_base",
    child="base_link",
    joint_type="prismatic",
    axis=[1, 0, 0],
    origin=origin_prismatic,
    limit=limit_prismatic,
    dynamics=None,
    safety_controller=None,
    calibration=None,
    mimic=None,
)

robot_prismatic.add_link(link_virtual_base_prismatic)
robot_prismatic.add_joint(joint_mobile_base_translation)

###############################################

# When specified, this sets more conservative joint limits than the
# original URDF. Joint limits that are outside the originally
# permitted range are clipped to the original range. Joint limits
# with a value of None are set to the original limit.
for robot in [robot_rotary, robot_prismatic]:
    for j in ik_joint_limits:
        joint = robot.joint_map.get(j, None)
        if joint is not None:

            original_upper = joint.limit.upper
            requested_upper = ik_joint_limits[j][1]
            # print()
            # print('joint =', j)
            # print('original_upper =', original_upper)
            # print('requested_upper =', requested_upper)
            if requested_upper is not None:
                new_upper = min(requested_upper, original_upper)
                # print('new_upper =', new_upper)
                robot.joint_map[j].limit.upper = new_upper
                # print()

            original_lower = joint.limit.lower
            requested_lower = ik_joint_limits[j][0]
            if requested_lower is not None:
                new_lower = max(requested_lower, original_lower)
                robot.joint_map[j].limit.lower = new_lower


# print('************************************************')
# print('after adding link and joint: robot =', robot)
# print('************************************************')

print()
save_urdf_file(robot_rotary, "stretch_base_rotation_ik.urdf")
save_urdf_file(robot_prismatic, "stretch_base_translation_ik.urdf")


# Create versions with fixed wrists

for robot in [robot_rotary, robot_prismatic]:
    print("Prepare URDF with a fixed wrist.")
    non_fixed_joints = [
        "joint_mobile_base_translation",
        "joint_mobile_base_rotation",
        "joint_lift",
        "joint_arm_l0",
    ]

    # Change any joint that should be immobile for end effector IK into a fixed joint
    for j in robot.joint_map.keys():
        if j not in non_fixed_joints:
            joint = robot.joint_map[j]
            # print('(joint name, joint type) =', (joint.name, joint.type))
            joint.type = "fixed"

save_urdf_file(robot_rotary, "stretch_base_rotation_ik_with_fixed_wrist.urdf")
save_urdf_file(robot_prismatic, "stretch_base_translation_ik_with_fixed_wrist.urdf")
