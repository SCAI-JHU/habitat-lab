from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pinocchio
from scipy.spatial.transform import Rotation as R

from habitat.articulated_agents.robots import StretchRobot
from habitat.core.logging import logger

POS_ERROR_TOL = 0.005
ORI_ERROR_TOL = [0.1, 0.1, np.pi / 2]

CEM_MAX_ITERATIONS = 5
CEM_NUM_SAMPLES = 50
CEM_NUM_TOP = 10

stretch_urdf_path = "/home/ubuntu/partnr-planner/third_party/habitat-lab/habitat-lab/habitat/articulated_agents/stretch_base_translation_ik.urdf"
# stretch_urdf_path = "/home/ubuntu/partnr-planner/data/robots/hab_stretch/urdf/hab_stretch.urdf"
# i = 0
# for f in model.frames:
#     print(i, f.name)
#     i+=1
# print("zzz",len(model.frames),model.frames[36])
default_controlled_links = [
    "joint_mobile_base_translation",
    "joint_arm_l0",
    "joint_arm_l1",
    "joint_arm_l2",
    "joint_arm_l3",
    "joint_lift",
    "joint_wrist_yaw",
    "joint_wrist_pitch",
    "joint_wrist_roll",
    # "joint_head_pan",
    # "joint_head_tilt",
]


class PinocchioIKSolver(StretchRobot):  # link_gripper_finger_left
    EPS = 1e-5
    DT = 1e-1
    DAMP = 1e-13

    def __init__(
        self,
        urdf_path: str = stretch_urdf_path,
        ee_link_name: str = "joint_lift",
        controlled_joints: List[str] = default_controlled_links,
        verbose: bool = False,
    ):
        # super().__init__(None, None)
        self.model = pinocchio.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.q_neutral = pinocchio.neutral(self.model)
        self.ee_frame_idx = [f.name for f in self.model.frames].index(ee_link_name)

        self.controlled_joints_by_name = {}
        self.controlled_joints = []
        self.controlled_joint_names = controlled_joints
        for joint in controlled_joints:
            if joint == "ignore":
                idx = -1
            else:
                jid = self.model.getJointId(joint)
                if jid >= len(self.model.idx_qs):
                    logger.error(f"{joint=} {jid=} not in model.idx_qs")
                    raise RuntimeError(
                        f"Invalid urdf at {urdf_path=}: missing {joint=}"
                    )
                else:
                    idx = self.model.idx_qs[jid]
            self.controlled_joints.append(idx)
            self.controlled_joints_by_name[joint] = idx

        logger.info(f"{controlled_joints=}")
        for j in controlled_joints:
            idx = self.model.getJointId(j)
            idx_q = self.model.idx_qs[idx]
            logger.info(f"{j=} {idx=} {idx_q=}")

    def get_dof(self) -> int:
        return len(self.controlled_joints)

    def get_num_controllable_joint(self) -> int:
        return len(self.controlled_joints)

    def _qmap_control2model(
        self, q_input: Union[np.ndarray, dict], ignore_missing_joints: bool = False
    ) -> np.ndarray:
        """returns a full joint configuration from a partial joint configuration"""
        q_out = self.q_neutral.copy()
        if isinstance(q_input, dict):
            for joint_name, value in q_input.items():
                if joint_name in self.controlled_joints_by_name:
                    q_out[self.controlled_joints_by_name[joint_name]] = value
                else:
                    jid = self.model.getJointId(joint_name)
                    if jid >= len(self.model.idx_qs):
                        if not ignore_missing_joints:
                            logger.error(
                                f"ERROR: {joint_name=} {jid=} not in model.idx_qs"
                            )
                            raise RuntimeError(
                                f"Tried to set joint not in model.idx_qs: {joint_name=}"
                            )
                    else:
                        q_out[self.model.idx_qs[self.model.getJointId(joint_name)]] = (
                            value
                        )
        else:
            assert len(self.controlled_joints) == len(
                q_input
            ), "if not specifying by name, must match length"
            for i, joint_idx in enumerate(self.controlled_joints):
                q_out[joint_idx] = q_input[i]
        return q_out

    def _qmap_model2control(self, q_input: np.ndarray) -> np.ndarray:
        """returns a partial joint configuration from a full joint configuration"""
        q_out = np.empty(len(self.controlled_joints))
        for i, joint_idx in enumerate(self.controlled_joints):
            if joint_idx >= 0:
                q_out[i] = q_input[joint_idx]

        return q_out

    def get_frame_pose(
        self,
        config: Union[np.ndarray, dict],
        node_a: str,
        node_b: str,
        ignore_missing_joints: bool = False,
    ) -> np.ndarray:
        """
        Get a transformation matrix transforming from node_a frame to node_b frame

        Args:
            config: joint values
            node_a: name of the first node
            node_b: name of the second node
            ignore_missing_joints: whether to ignore missing joints in the configuration

        Returns:
            transformation matrix from node_a to node_b
        """
        q_model = self._qmap_control2model(
            config, ignore_missing_joints=ignore_missing_joints
        )
        # print('q_model', q_model)
        pinocchio.forwardKinematics(self.model, self.data, q_model)
        frame_idx1 = [f.name for f in self.model.frames].index(node_a)
        frame_idx2 = [f.name for f in self.model.frames].index(node_b)
        # print(frame_idx1)
        # print(frame_idx2)
        # print(self.model.getFrameId(node_a))
        # print(self.model.getFrameId(node_b))
        # frame_idx1 = self.model.getFrameId(node_a)
        # frame_idx2 = self.model.getFrameId(node_b)
        pinocchio.updateFramePlacement(self.model, self.data, frame_idx1)
        placement_frame1 = self.data.oMf[frame_idx1]
        pinocchio.updateFramePlacement(self.model, self.data, frame_idx2)
        placement_frame2 = self.data.oMf[frame_idx2]
        # print('pin 1', placement_frame1)
        # print('pin 2', placement_frame2)
        return placement_frame2.inverse() * placement_frame1

    def compute_fk(
        self,
        config: np.ndarray,
        link_name: str = None,
        ignore_missing_joints: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        base_frame_name = "link_arm_l0"
        base_frame_idx = self.model.getFrameId(base_frame_name)

        base = self.data.oMf[base_frame_idx]
        """Given joint values, return end-effector position and quaternion associated with it.

        Args:
            config: joint values
            link_name: name of the link to compute FK for; if None, uses the end-effector link

        Returns:
            pos: end-effector position (x, y, z)
            quat: end-effector quaternion (w, x, y, z)
        """
        if link_name is None:
            frame_idx = self.ee_frame_idx
        else:
            try:
                frame_idx = [f.name for f in self.model.frames].index(link_name)
            except ValueError:
                logger.error(
                    f"Unknown link_name {link_name}. Defaulting to end-effector"
                )
                frame_idx = self.ee_frame_idx
        q_model = self._qmap_control2model(
            config, ignore_missing_joints=ignore_missing_joints
        )
        pinocchio.forwardKinematics(self.model, self.data, q_model)
        pinocchio.updateFramePlacement(self.model, self.data, frame_idx)
        pos = self.data.oMf[frame_idx].translation
        quat = R.from_matrix(self.data.oMf[frame_idx].rotation).as_quat()
        return pos.copy(), quat.copy()

    def compute_ik(
        self,
        pos_desired: np.ndarray,
        quat_desired: np.ndarray = np.array([1, 0, 0, 0]),
        q_init=None,
        max_iterations=300,
        num_attempts: int = 1,
        verbose: bool = False,
        ignore_missing_joints: bool = False,
        custom_ee_frame: Optional[str] = None,
    ) -> Tuple[np.ndarray, bool, dict]:
        """given end-effector position and quaternion, return joint values.

        Two parameters are currently unused and might be implemented in the future:
            q_init: initial configuration for the optimization to start in; especially useful for
                    arms with redundant degrees of freedom
            num_attempts: start from multiple initial configs; included for compatibility with pb
            max iterations: time budget in number of steps; included for compatibility with pb
        """
        i = 0
        if custom_ee_frame is not None:
            _ee_frame_idx = [f.name for f in self.model.frames].index(custom_ee_frame)
        else:
            _ee_frame_idx = self.ee_frame_idx

        if q_init is None:
            q = self.q_neutral.copy()
            if num_attempts > 1:
                raise NotImplementedError(
                    "Sampling multiple initial configs not yet supported by Pinocchio solver."
                )
        else:
            q = self._qmap_control2model(
                q_init, ignore_missing_joints=ignore_missing_joints
            )
            # Override the number of attempts
            num_attempts = 1

        desired_ee_pose = pinocchio.SE3(
            R.from_quat(quat_desired).as_matrix(), pos_desired
        )
        #        lower_limits, upper_limits = self.get_joint_limits()
        while True:
            pinocchio.forwardKinematics(self.model, self.data, q)
            pinocchio.updateFramePlacement(self.model, self.data, _ee_frame_idx)
            dMi = desired_ee_pose.actInv(self.data.oMf[_ee_frame_idx])
            # print("self.data.oMf[_ee_frame_idx]",self.data.oMf[_ee_frame_idx])
            err = pinocchio.log(dMi).vector
            # pos_err = err[3:6]

            if verbose:
                print(f"[pinocchio_ik_solver] iter={i}; error={err}")
            pos_desired = desired_ee_pose.translation
            pos_current = self.data.oMf[_ee_frame_idx].translation
            pos_err = pos_desired - pos_current
            if np.linalg.norm(pos_err) < self.EPS:
                success = True
                break
            if i >= max_iterations:
                success = False
                break
            J = pinocchio.computeFrameJacobian(
                self.model,
                self.data,
                q,
                _ee_frame_idx,
                pinocchio.ReferenceFrame.LOCAL,
            )
            v = -J.T.dot(np.linalg.solve(J.dot(J.T) + self.DAMP * np.eye(6), err))
            q = pinocchio.integrate(self.model, q, v * self.DT)
            #   q = np.clip(q, lower_limits, upper_limits)
            i += 1

        q_control = self._qmap_model2control(q.flatten())
        debug_info = {"iter": i, "pos_err": err}
        print(debug_info)

        base_frame_name = "link_gripper_finger_left"
        base_frame_idx = self.model.getFrameId(base_frame_name)

        base = self.data.oMf[base_frame_idx]
        # print("basexytest",base)

        return q_control, success, debug_info

    def q_array_to_dict(self, arr: np.ndarray):
        state = {}
        assert len(arr) == len(self.controlled_joint_names)
        for i, name in enumerate(self.controlled_joint_names):
            state[name] = arr[i]
        return state

    def set_initail_joint_state(self, joint_state: List):
        initial_joint_state = {}
        for i in range(len(default_controlled_links)):
            initial_joint_state[default_controlled_links[i]] = joint_state[i]
        return initial_joint_state


manip_ik_solver = PinocchioIKSolver()

initial_joint_state = {
    "joint_mobile_base_translation": 0.0,
    "joint_arm_l0": 0.0,
    "joint_arm_l1": 0.0,
    "joint_arm_l2": 0.0,
    "joint_arm_l3": 0.0,
    "joint_lift": 0.0,
    "joint_wrist_yaw": 0.0,
    "joint_wrist_pitch": 0.0,
    "joint_wrist_roll": 0.0,
    # "joint_head_pan": 0.0,
    # "joint_head_tilt": 0.0,
}
joint_state = [0, 0, 0, 0, 0, 0, 0, 0, 0]


ee_pose = manip_ik_solver.compute_fk(initial_joint_state)
# print(f"{ee_pose=}")
assert ee_pose is not None, "FK failed"


# rotation_matrix = np.array([
#     [-0.514521, 3.67323e-06, -0.857478],
#     [0.857478, 7.34638e-06, -0.514521],
#     [4.40941e-06, -1, -6.92958e-06]
# ])

# rotation = R.from_matrix(rotation_matrix)
# quaternion = rotation.as_quat()
# print("quaternion",quaternion)

#     # Test Inverse Kinematics
# ee_position = np.array([-0.03, -0.4, 0.9])
# ee_orientation = np.array([0, 0, 0, 1])
# initial_joint_state = manip_ik_solver.set_initail_joint_state(joint_state)
# print(initial_joint_state)
# res, success, info = manip_ik_solver.compute_ik(
#         ee_position,
#         ee_orientation,
#         q_init=initial_joint_state,
#     )
# print("Result =", res)
# print("Success =", success)
# assert success, "IK failed"

# #     # Test IK accuracy
# initial_joint_state = manip_ik_solver.set_initail_joint_state(res)
# res_ee_position, res_ee_orientation = manip_ik_solver.compute_fk(initial_joint_state)
# print("res_ee_position",res_ee_position)
# ee_position_error = np.linalg.norm(res_ee_position - ee_position)
# ee_orientation_error = np.linalg.norm(res_ee_orientation - ee_orientation)
# EPS_IK_CORRECT = 0.05
# assert ee_position_error < EPS_IK_CORRECT, "IK position error too large"
# assert ee_orientation_error < EPS_IK_CORRECT, "IK orientation error too large"
