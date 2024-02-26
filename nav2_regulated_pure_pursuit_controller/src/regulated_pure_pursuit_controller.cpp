// Copyright (c) 2020 Shrijit Singh
// Copyright (c) 2020 Samsung Research America
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <string>
#include <limits>
#include <memory>
#include <utility>

#include "nav2_regulated_pure_pursuit_controller/regulated_pure_pursuit_controller.hpp"
#include "nav2_core/exceptions.hpp"
#include "nav2_util/node_utils.hpp"
#include "nav2_util/geometry_utils.hpp"
#include "nav2_costmap_2d/costmap_filters/filter_values.hpp"

using std::hypot;
using std::min;
using std::max;
using std::abs;
using nav2_util::declare_parameter_if_not_declared;
using nav2_util::geometry_utils::euclidean_distance;
using namespace nav2_costmap_2d;  // NOLINT
using namespace std::chrono_literals;  // NOLINT

namespace nav2_regulated_pure_pursuit_controller
{

void RegulatedPurePursuitController::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
  std::string name, const std::shared_ptr<tf2_ros::Buffer> & tf,
  const std::shared_ptr<nav2_costmap_2d::Costmap2DROS> & costmap_ros)
{
  auto node = parent.lock();
  if (!node) {
    throw nav2_core::PlannerException("Unable to lock node!");
  }

  costmap_ros_ = costmap_ros;
  costmap_ = costmap_ros_->getCostmap();
  tf_ = tf;
  plugin_name_ = name;
  logger_ = node->get_logger();
  clock_ = node->get_clock();

  double transform_tolerance = 0.1;
  double control_frequency = 20.0;
  goal_dist_tol_ = 0.25;  // reasonable default before first update

  declare_parameter_if_not_declared(
    node, plugin_name_ + ".desired_linear_vel", rclcpp::ParameterValue(0.5));
  declare_parameter_if_not_declared(
    node, plugin_name_ + ".lookahead_dist", rclcpp::ParameterValue(0.6));
  declare_parameter_if_not_declared(
    node, plugin_name_ + ".min_lookahead_dist", rclcpp::ParameterValue(0.3));
  declare_parameter_if_not_declared(
    node, plugin_name_ + ".max_lookahead_dist", rclcpp::ParameterValue(0.9));
  declare_parameter_if_not_declared(
    node, plugin_name_ + ".lookahead_time", rclcpp::ParameterValue(1.5));
  declare_parameter_if_not_declared(
    node, plugin_name_ + ".rotate_to_heading_angular_vel", rclcpp::ParameterValue(1.8));
  declare_parameter_if_not_declared(
    node, plugin_name_ + ".transform_tolerance", rclcpp::ParameterValue(0.1));
  declare_parameter_if_not_declared(
    node, plugin_name_ + ".use_velocity_scaled_lookahead_dist",
    rclcpp::ParameterValue(false));
  declare_parameter_if_not_declared(
    node, plugin_name_ + ".min_approach_linear_velocity", rclcpp::ParameterValue(0.05));
  declare_parameter_if_not_declared(
    node, plugin_name_ + ".use_approach_linear_velocity_scaling", rclcpp::ParameterValue(true));
  declare_parameter_if_not_declared(
    node, plugin_name_ + ".max_allowed_time_to_collision", rclcpp::ParameterValue(1.0));
  declare_parameter_if_not_declared(
    node, plugin_name_ + ".use_regulated_linear_velocity_scaling", rclcpp::ParameterValue(true));
  declare_parameter_if_not_declared(
    node, plugin_name_ + ".use_cost_regulated_linear_velocity_scaling",
    rclcpp::ParameterValue(true));
  declare_parameter_if_not_declared(
    node, plugin_name_ + ".cost_scaling_dist", rclcpp::ParameterValue(0.6));
  declare_parameter_if_not_declared(
    node, plugin_name_ + ".cost_scaling_gain", rclcpp::ParameterValue(1.0));
  declare_parameter_if_not_declared(
    node, plugin_name_ + ".inflation_cost_scaling_factor", rclcpp::ParameterValue(3.0));
  declare_parameter_if_not_declared(
    node, plugin_name_ + ".regulated_linear_scaling_min_radius", rclcpp::ParameterValue(0.90));
  declare_parameter_if_not_declared(
    node, plugin_name_ + ".regulated_linear_scaling_min_speed", rclcpp::ParameterValue(0.25));
  declare_parameter_if_not_declared(
    node, plugin_name_ + ".use_rotate_to_heading", rclcpp::ParameterValue(true));
  declare_parameter_if_not_declared(
    node, plugin_name_ + ".rotate_to_heading_min_angle", rclcpp::ParameterValue(0.785));
  declare_parameter_if_not_declared(
    node, plugin_name_ + ".max_angular_accel", rclcpp::ParameterValue(3.2));
  declare_parameter_if_not_declared(
    node, plugin_name_ + ".allow_reversing", rclcpp::ParameterValue(false));

  node->get_parameter(plugin_name_ + ".desired_linear_vel", desired_linear_vel_);
  base_desired_linear_vel_ = desired_linear_vel_;
  node->get_parameter(plugin_name_ + ".lookahead_dist", lookahead_dist_);
  node->get_parameter(plugin_name_ + ".min_lookahead_dist", min_lookahead_dist_);
  node->get_parameter(plugin_name_ + ".max_lookahead_dist", max_lookahead_dist_);
  node->get_parameter(plugin_name_ + ".lookahead_time", lookahead_time_);
  node->get_parameter(
    plugin_name_ + ".rotate_to_heading_angular_vel",
    rotate_to_heading_angular_vel_);
  node->get_parameter(plugin_name_ + ".transform_tolerance", transform_tolerance);
  node->get_parameter(
    plugin_name_ + ".use_velocity_scaled_lookahead_dist",
    use_velocity_scaled_lookahead_dist_);
  node->get_parameter(
    plugin_name_ + ".min_approach_linear_velocity",
    min_approach_linear_velocity_);
  node->get_parameter(
    plugin_name_ + ".use_approach_linear_velocity_scaling",
    use_approach_vel_scaling_);
  node->get_parameter(
    plugin_name_ + ".max_allowed_time_to_collision",
    max_allowed_time_to_collision_);
  node->get_parameter(
    plugin_name_ + ".use_regulated_linear_velocity_scaling",
    use_regulated_linear_velocity_scaling_);
  node->get_parameter(
    plugin_name_ + ".use_cost_regulated_linear_velocity_scaling",
    use_cost_regulated_linear_velocity_scaling_);
  node->get_parameter(plugin_name_ + ".cost_scaling_dist", cost_scaling_dist_);
  node->get_parameter(plugin_name_ + ".cost_scaling_gain", cost_scaling_gain_);
  node->get_parameter(
    plugin_name_ + ".inflation_cost_scaling_factor",
    inflation_cost_scaling_factor_);
  node->get_parameter(
    plugin_name_ + ".regulated_linear_scaling_min_radius",
    regulated_linear_scaling_min_radius_);
  node->get_parameter(
    plugin_name_ + ".regulated_linear_scaling_min_speed",
    regulated_linear_scaling_min_speed_);
  node->get_parameter(plugin_name_ + ".use_rotate_to_heading", use_rotate_to_heading_);
  node->get_parameter(plugin_name_ + ".rotate_to_heading_min_angle", rotate_to_heading_min_angle_);
  node->get_parameter(plugin_name_ + ".max_angular_accel", max_angular_accel_);
  node->get_parameter(plugin_name_ + ".allow_reversing", allow_reversing_);
  node->get_parameter("controller_frequency", control_frequency);

  transform_tolerance_ = tf2::durationFromSec(transform_tolerance);
  control_duration_ = 1.0 / control_frequency;

  if (inflation_cost_scaling_factor_ <= 0.0) {
    RCLCPP_WARN(
      logger_, "The value inflation_cost_scaling_factor is incorrectly set, "
      "it should be >0. Disabling cost regulated linear velocity scaling.");
    use_cost_regulated_linear_velocity_scaling_ = false;
  }

  /** Possible to drive in reverse direction if and only if
   "use_rotate_to_heading" parameter is set to false **/

  if (use_rotate_to_heading_ && allow_reversing_) {
    RCLCPP_WARN(
      logger_, "Disabling reversing. Both use_rotate_to_heading and allow_reversing "
      "parameter cannot be set to true. By default setting use_rotate_to_heading true");
    allow_reversing_ = false;
  }

  global_path_pub_ = node->create_publisher<nav_msgs::msg::Path>("received_global_plan", 1);
  carrot_pub_ = node->create_publisher<geometry_msgs::msg::PointStamped>("lookahead_point", 1);
  closest_pub_ = node->create_publisher<geometry_msgs::msg::PointStamped>("closest_point", 1);
  carrot_arc_pub_ = node->create_publisher<nav_msgs::msg::Path>("lookahead_collision_arc", 1);
}

void RegulatedPurePursuitController::cleanup()
{
  RCLCPP_INFO(
    logger_,
    "Cleaning up controller: %s of type"
    " regulated_pure_pursuit_controller::RegulatedPurePursuitController",
    plugin_name_.c_str());
  global_path_pub_.reset();
  carrot_pub_.reset();
  closest_pub_.reset();
  carrot_arc_pub_.reset();
}

void RegulatedPurePursuitController::activate()
{
  RCLCPP_INFO(
    logger_,
    "Activating controller: %s of type "
    "regulated_pure_pursuit_controller::RegulatedPurePursuitController",
    plugin_name_.c_str());
  global_path_pub_->on_activate();
  carrot_pub_->on_activate();
  closest_pub_->on_activate();
  carrot_arc_pub_->on_activate();
}

void RegulatedPurePursuitController::deactivate()
{
  RCLCPP_INFO(
    logger_,
    "Deactivating controller: %s of type "
    "regulated_pure_pursuit_controller::RegulatedPurePursuitController",
    plugin_name_.c_str());
  global_path_pub_->on_deactivate();
  carrot_pub_->on_deactivate();
  closest_pub_->on_deactivate();
  carrot_arc_pub_->on_deactivate();
}

std::unique_ptr<geometry_msgs::msg::PointStamped> RegulatedPurePursuitController::createCarrotMsg(
  const geometry_msgs::msg::PoseStamped & carrot_pose)
{
  auto carrot_msg = std::make_unique<geometry_msgs::msg::PointStamped>();
  carrot_msg->header = carrot_pose.header;
  carrot_msg->point.x = carrot_pose.pose.position.x;
  carrot_msg->point.y = carrot_pose.pose.position.y;
  carrot_msg->point.z = 0.01;  // publish right over map to stand out
  return carrot_msg;
}

double RegulatedPurePursuitController::getLookAheadDistance(const geometry_msgs::msg::Twist & speed)
{
  // If using velocity-scaled look ahead distances, find and clamp the dist
  // Else, use the static look ahead distance
  double lookahead_dist = lookahead_dist_;
  if (use_velocity_scaled_lookahead_dist_) {
    lookahead_dist = fabs(speed.linear.x) * lookahead_time_;
    lookahead_dist = std::clamp(lookahead_dist, min_lookahead_dist_, max_lookahead_dist_);
  }

  return lookahead_dist;
}

geometry_msgs::msg::TwistStamped RegulatedPurePursuitController::computeVelocityCommands(
  const geometry_msgs::msg::PoseStamped & pose,
  const geometry_msgs::msg::Twist & speed,
  nav2_core::GoalChecker * goal_checker)
{
  // Update for the current goal checker's state
  geometry_msgs::msg::Pose pose_tolerance;
  geometry_msgs::msg::Twist vel_tolerance;
  if (!goal_checker->getTolerances(pose_tolerance, vel_tolerance)) {
    RCLCPP_WARN(logger_, "Unable to retrieve goal checker's tolerances!");
  } else {
    goal_dist_tol_ = pose_tolerance.position.x;
  }

  // Transform path to robot base frame
  double distance_to_goal;
  auto transformed_plan = transformGlobalPlan(pose, distance_to_goal);

  // Find look ahead distance and point on path and publish
  double lookahead_dist = getLookAheadDistance(speed);

  // Check for reverse driving
  if (allow_reversing_) {
    // Cusp check
    double dist_to_direction_change = findDirectionChange(pose);

    // if the lookahead distance is further than the cusp, use the cusp distance instead
    if (dist_to_direction_change < lookahead_dist) {
      lookahead_dist = dist_to_direction_change;
    }
  }

  auto carrot_pose = getLookAheadPoint(lookahead_dist, transformed_plan);
  carrot_pub_->publish(createCarrotMsg(carrot_pose));

  float deviation_from_path = std::numeric_limits<float>::max();
  bool overshoot_goal = false;
  auto closes_point = getLookAheadPoint(0.0, transformed_plan, deviation_from_path, overshoot_goal);
  closest_pub_->publish(createCarrotMsg(closes_point));

  if (deviation_from_path < MAXIMUM_DEVIATION_FROM_PATH) {
    last_on_track_ts_ = std::chrono::system_clock::now();
  }

  if (!overshoot_goal) {
    last_no_overshoot_ts_ = std::chrono::system_clock::now();
  }

  if (
    std::chrono::duration_cast<std::chrono::seconds>(
      std::chrono::system_clock::now() - last_on_track_ts_)
      .count() > MAXIMUM_ALLOWED_TIME_OFF_TRACK) {
    std::ostringstream error_msg;
    error_msg << "RegulatedPurePursuitController detected that the robot has deviated more than ";
    error_msg << MAXIMUM_DEVIATION_FROM_PATH << " meters from its path for more than ";
    error_msg << MAXIMUM_ALLOWED_TIME_OFF_TRACK << " seconds!";
    throw nav2_core::PlannerException(error_msg.str());
  }

  if (
    std::chrono::duration_cast<std::chrono::seconds>(
      std::chrono::system_clock::now() - last_no_overshoot_ts_)
      .count() > MAXIMUM_ALLOWED_TIME_OVERSHOOT) {
    std::ostringstream error_msg;
    error_msg << "RegulatedPurePursuitController detected that the robot has overshoot its ";
    error_msg << "goal by " << MAXIMUM_OVERSHOOT << " meters for more than ";
    error_msg << MAXIMUM_ALLOWED_TIME_OVERSHOOT << " seconds!";
    throw nav2_core::PlannerException(error_msg.str());
  }

  double linear_vel, angular_vel;

  // Find distance^2 to look ahead point (carrot) in robot base frame
  // This is the chord length of the circle
  const double carrot_dist2 =
    (carrot_pose.pose.position.x * carrot_pose.pose.position.x) +
    (carrot_pose.pose.position.y * carrot_pose.pose.position.y);

  // Find curvature of circle (k = 1 / R)
  double curvature = 0.0;
  if (carrot_dist2 > 0.001) {
    curvature = 2.0 * carrot_pose.pose.position.y / carrot_dist2;
    if (abs(curvature) > (1 / MINIMUM_TURNING_RADIUS)) {
      const double sign = curvature < 0.0 ? -1.0 : 1.0;
      curvature = (1 / MINIMUM_TURNING_RADIUS) * sign;
    }
  }

  // Setting the velocity direction
  double sign = 1.0;
  if (allow_reversing_) {
    sign = carrot_pose.pose.position.x >= 0.0 ? 1.0 : -1.0;
  }

  linear_vel = desired_linear_vel_;

  // Make sure we're in compliance with basic constraints
  // double angle_to_heading;
  // if (shouldRotateToGoalHeading(carrot_pose)) {
  //   double angle_to_goal = tf2::getYaw(transformed_plan.poses.back().pose.orientation);
  //   rotateToHeading(linear_vel, angular_vel, angle_to_goal, speed);
  // } else if (shouldRotateToPath(carrot_pose, angle_to_heading)) {
  //   rotateToHeading(linear_vel, angular_vel, angle_to_heading, speed);
  // } else {
    applyConstraints(
      distance_to_goal,
      lookahead_dist, curvature, speed,
      deviation_from_path,
      costAtPose(pose.pose.position.x, pose.pose.position.y), linear_vel, sign);

    // Apply curvature to angular velocity after constraining linear velocity
    angular_vel = linear_vel * curvature;
  // }

  // Collision checking on this velocity heading
  if (isCollisionImminent(pose, linear_vel, angular_vel)) {
    throw nav2_core::PlannerException("RegulatedPurePursuitController detected collision ahead!");
  }

  // When we only have one pose left it means that we are near the end of the path
  if (transformed_plan.poses.size() == 1) {
    double x = transformed_plan.poses.at(0).pose.position.x;
    // The transformed plan is in the robot frame so we can look where the goal is in the
    // forward (x) direction.
    if (x < -goal_dist_tol_) {
      // If we have gone past the goal this far there is not sane to recover, instead fail
      std::ostringstream error_msg;
      error_msg << "RegulatedPurePursuitController detected that the robot has overshot its goal";
      throw nav2_core::PlannerException(error_msg.str());
    } else if (x < 0) {
      // If we are in front of the goal we need to back up a bit
      linear_vel = -min_approach_linear_velocity_;
    }
  }

  // populate and return message
  geometry_msgs::msg::TwistStamped cmd_vel;
  cmd_vel.header = pose.header;
  cmd_vel.twist.linear.x = linear_vel;
  cmd_vel.twist.angular.z = angular_vel;
  return cmd_vel;
}

bool RegulatedPurePursuitController::shouldRotateToPath(
  const geometry_msgs::msg::PoseStamped & carrot_pose, double & angle_to_path)
{
  // Whether we should rotate robot to rough path heading
  angle_to_path = atan2(carrot_pose.pose.position.y, carrot_pose.pose.position.x);
  return use_rotate_to_heading_ && fabs(angle_to_path) > rotate_to_heading_min_angle_;
}

bool RegulatedPurePursuitController::shouldRotateToGoalHeading(
  const geometry_msgs::msg::PoseStamped & carrot_pose)
{
  // Whether we should rotate robot to goal heading
  double dist_to_goal = std::hypot(carrot_pose.pose.position.x, carrot_pose.pose.position.y);
  return use_rotate_to_heading_ && dist_to_goal < goal_dist_tol_;
}

void RegulatedPurePursuitController::rotateToHeading(
  double & linear_vel, double & angular_vel,
  const double & angle_to_path, const geometry_msgs::msg::Twist & curr_speed)
{
  // Rotate in place using max angular velocity / acceleration possible
  linear_vel = 0.0;
  const double sign = angle_to_path > 0.0 ? 1.0 : -1.0;
  angular_vel = sign * rotate_to_heading_angular_vel_;

  const double & dt = control_duration_;
  const double min_feasible_angular_speed = curr_speed.angular.z - max_angular_accel_ * dt;
  const double max_feasible_angular_speed = curr_speed.angular.z + max_angular_accel_ * dt;
  angular_vel = std::clamp(angular_vel, min_feasible_angular_speed, max_feasible_angular_speed);
}

geometry_msgs::msg::PoseStamped RegulatedPurePursuitController::getLookAheadPoint(
  const double & lookahead_dist,
  const nav_msgs::msg::Path & transformed_plan)
{
  float dont_care_1 = std::numeric_limits<float>::max();
  bool dont_care_2 = false;
  return getLookAheadPoint(lookahead_dist, transformed_plan, dont_care_1, dont_care_2);
}

geometry_msgs::msg::PoseStamped RegulatedPurePursuitController::getLookAheadPoint(
  const double & lookahead_dist, const nav_msgs::msg::Path & transformed_plan,
  float & deviation_from_path, bool & overshoot_goal)
{
  // Find the first pose which is at a distance greater than the lookahead distance
  // -> and also in front of the robot!
  auto b_pose_it = std::find_if(
    transformed_plan.poses.begin(), transformed_plan.poses.end(), [&](const auto & ps) {
      return hypot(ps.pose.position.x, ps.pose.position.y) >= lookahead_dist &&
             ps.pose.position.x >= 0;
    });

  // If the no pose is not far enough, take the last pose
  if (b_pose_it == transformed_plan.poses.end()) {
    b_pose_it = std::prev(transformed_plan.poses.end());
    // RCLCPP_INFO(logger_, "Using last pose");
    geometry_msgs::msg::PoseStamped res = *b_pose_it;
    deviation_from_path =
      std::sqrt(std::pow(res.pose.position.x, 2) + std::pow(res.pose.position.y, 2));

    if (res.pose.position.x < -MAXIMUM_OVERSHOOT) {
      overshoot_goal = true;
    }

    // If the last pose is closer than the lookahead distance we push it forward along the x axis
    // of the last pose until it is approximately one lookahead distance away from the robot. We
    // do this because otherwise the robot can turn too sharply when it gets close to the goal.
    tf2::Transform res_pose_tf;
    tf2::fromMsg(res.pose, res_pose_tf);
    tf2::Transform res_pose_tf_cpy = res_pose_tf;
    // Remove the rotation from the goal pose so that x points forward
    res_pose_tf_cpy.setOrigin(tf2::Vector3(0.0, 0.0, 0.0));
    res_pose_tf = res_pose_tf_cpy.inverseTimes(res_pose_tf);
    // Push forward in the x forward direction
    res_pose_tf.getOrigin().setX(
      res_pose_tf.getOrigin().x() + lookahead_dist -
      KDL::sign(res.pose.position.x) * hypot(res.pose.position.x, res.pose.position.y));
    // Add back the rotation that we removed earlier
    res_pose_tf = res_pose_tf_cpy * res_pose_tf;
    tf2::toMsg(res_pose_tf, res.pose);
    // std::cout << "last x " << res.pose.position.x << " last y "
    // << res.pose.position.y << std::endl;
    return res;
  }

  // Special case if the pose is the first pose in the plan
  if (b_pose_it == transformed_plan.poses.begin()) {
    // RCLCPP_INFO(logger_, "Using first pose");
    auto pose = *b_pose_it;
    deviation_from_path =
      std::sqrt(std::pow(pose.pose.position.x, 2) + std::pow(pose.pose.position.y, 2));
    // std::cout << "first x " << pose.pose.position.x << " first y " << pose.pose.position.y
    //           << std::endl;
    return pose;
  }

  // If it's not the first pose then we interpolate with the previous pose
  auto a_pose_it = std::prev(b_pose_it);

  auto a_pose = *a_pose_it;
  auto b_pose = *b_pose_it;

  // https://math.stackexchange.com/questions/422602/convert-two-points-to-line-eq-ax-by-c-0
  float A = a_pose.pose.position.y - b_pose.pose.position.y;
  float B = b_pose.pose.position.x - a_pose.pose.position.x;
  float C = a_pose.pose.position.x * b_pose.pose.position.y -
            b_pose.pose.position.x * a_pose.pose.position.y;

  // https://cp-algorithms.com/geometry/circle-line-intersection.html
  float d0 = fabs(C) / sqrt(A * A + B * B);
  // closest point on line
  float x0 = -A * C / (A * A + B * B);
  float y0 = -B * C / (A * A + B * B);

  // zero or one intersection
  float carrot_x = x0 + B * lookahead_dist;
  float carrot_y = y0 - A * lookahead_dist;
  // two intersections
  if (d0 < lookahead_dist) {
    float d = sqrt(lookahead_dist * lookahead_dist - C * C / (A * A + B * B));
    float m = sqrt(d * d / (A * A + B * B));
    carrot_x = x0 + B * m;
    carrot_y = y0 - A * m;
    // std::cout << "A * A + B * B = " << A * A + B * B << " C = " << C << " d0 = " << d0
    // << " A = " << A << " B = " << B << " d = " << d << " m = " << m << std::endl;
  } else {
    // std::cout << "A * A + B * B = " << A * A + B * B << " C = " << C << " d0 = " << d0
    //           << " A = " << A << " B = " << B << std::endl;
  }

  // std::cout << "carrot_x " << carrot_x << " carrot_y " << carrot_y << std::endl;

  geometry_msgs::msg::PoseStamped carrot_pose;
  carrot_pose.header.frame_id = b_pose.header.frame_id;
  carrot_pose.header.stamp = b_pose.header.stamp;
  carrot_pose.pose.position.x = carrot_x;
  carrot_pose.pose.position.y = carrot_y;

  deviation_from_path = sqrt(x0 * x0 + y0 * y0);
  return carrot_pose;
}

bool RegulatedPurePursuitController::isCollisionImminent(
  const geometry_msgs::msg::PoseStamped & robot_pose,
  const double & linear_vel, const double & angular_vel)
{
  // Note(stevemacenski): This may be a bit unusual, but the robot_pose is in
  // odom frame and the carrot_pose is in robot base frame.

  // check current point is OK
  if (inCollision(robot_pose.pose.position.x, robot_pose.pose.position.y)) {
    return true;
  }

  // visualization messages
  nav_msgs::msg::Path arc_pts_msg;
  arc_pts_msg.header.frame_id = costmap_ros_->getGlobalFrameID();
  arc_pts_msg.header.stamp = robot_pose.header.stamp;
  geometry_msgs::msg::PoseStamped pose_msg;
  pose_msg.header.frame_id = arc_pts_msg.header.frame_id;
  pose_msg.header.stamp = arc_pts_msg.header.stamp;

  geometry_msgs::msg::Pose2D curr_pose;
  curr_pose.x = robot_pose.pose.position.x;
  curr_pose.y = robot_pose.pose.position.y;
  curr_pose.theta = tf2::getYaw(robot_pose.pose.orientation);

  int i = 1;
  while (i * PROJECTION_TIME < max_allowed_time_to_collision_) {
    i++;

    // apply velocity at curr_pose over distance
    curr_pose.x += PROJECTION_TIME * (linear_vel * cos(curr_pose.theta));
    curr_pose.y += PROJECTION_TIME * (linear_vel * sin(curr_pose.theta));
    curr_pose.theta += PROJECTION_TIME * angular_vel;

    // store it for visualization
    pose_msg.pose.position.x = curr_pose.x;
    pose_msg.pose.position.y = curr_pose.y;
    pose_msg.pose.position.z = 0.01;
    arc_pts_msg.poses.push_back(pose_msg);

    // check for collision at this point
    if (inCollision(curr_pose.x, curr_pose.y)) {
      carrot_arc_pub_->publish(arc_pts_msg);
      return true;
    }
  }

  carrot_arc_pub_->publish(arc_pts_msg);

  return false;
}

bool RegulatedPurePursuitController::inCollision(const double & x, const double & y)
{
  unsigned int mx, my;

  if (!costmap_->worldToMap(x, y, mx, my)) {
    RCLCPP_WARN_THROTTLE(
      logger_, *(clock_), 30000,
      "The dimensions of the costmap is too small to successfully check for "
      "collisions as far ahead as requested. Proceed at your own risk, slow the robot, or "
      "increase your costmap size.");
    return false;
  }

  unsigned char cost = costmap_->getCost(mx, my);

  if (costmap_ros_->getLayeredCostmap()->isTrackingUnknown()) {
    return cost >= INSCRIBED_INFLATED_OBSTACLE && cost != NO_INFORMATION;
  } else {
    return cost >= INSCRIBED_INFLATED_OBSTACLE;
  }
}

double RegulatedPurePursuitController::costAtPose(const double & x, const double & y)
{
  unsigned int mx, my;

  if (!costmap_->worldToMap(x, y, mx, my)) {
    RCLCPP_FATAL(
      logger_,
      "The dimensions of the costmap is too small to fully include your robot's footprint, "
      "thusly the robot cannot proceed further");
    throw nav2_core::PlannerException(
            "RegulatedPurePursuitController: Dimensions of the costmap are too small "
            "to encapsulate the robot footprint at current speeds!");
  }

  unsigned char cost = costmap_->getCost(mx, my);
  return static_cast<double>(cost);
}

void RegulatedPurePursuitController::applyConstraints(
  const double & distance_to_goal, const double & /*lookahead_dist*/,
  const double & curvature, const geometry_msgs::msg::Twist & /*curr_speed*/,
  const double & deviation_from_path,
  const double & pose_cost, double & linear_vel, double & sign)
{
  double curvature_vel = linear_vel;
  double cost_vel = linear_vel;
  double approach_vel = linear_vel;
  double deviation_vel = linear_vel;

  // limit linear velocity by deviation from path
  deviation_vel *= 0.1 + 0.9 * 
    std::pow(
      std::clamp(1.0 - deviation_from_path / MAXIMUM_DEVIATION_FROM_PATH, 0.0, 1.0), 2);

  // limit the linear velocity by curvature
  const double radius = fabs(1.0 / curvature);
  const double & min_rad = regulated_linear_scaling_min_radius_;
  if (use_regulated_linear_velocity_scaling_ && radius < min_rad) {
    curvature_vel *= 1.0 - (fabs(radius - min_rad) / min_rad);
  }

  // limit the linear velocity by proximity to obstacles
  if (use_cost_regulated_linear_velocity_scaling_ &&
    pose_cost != static_cast<double>(NO_INFORMATION) &&
    pose_cost != static_cast<double>(FREE_SPACE))
  {
    const double inscribed_radius = costmap_ros_->getLayeredCostmap()->getInscribedRadius();
    const double min_distance_to_obstacle = (-1.0 / inflation_cost_scaling_factor_) *
      std::log(pose_cost / (INSCRIBED_INFLATED_OBSTACLE - 1)) + inscribed_radius;

    if (min_distance_to_obstacle < cost_scaling_dist_) {
      cost_vel *= cost_scaling_gain_ * min_distance_to_obstacle / cost_scaling_dist_;
    }
  }

  // Use the lowest of the 2 constraint heuristics, but above the minimum translational speed
  linear_vel = std::min({cost_vel, curvature_vel, deviation_vel});
  linear_vel = std::max(linear_vel, regulated_linear_scaling_min_speed_);

  // if the actual lookahead distance is shorter than requested, that means we're at the
  // end of the path. We'll scale linear velocity by error to slow to a smooth stop.
  // This expression is eq. to (1) holding time to goal, t, constant using the theoretical
  // lookahead distance and proposed velocity and (2) using t with the actual lookahead
  // distance to scale the velocity (e.g. t = lookahead / velocity, v = carrot / t).
  if (distance_to_goal < NEAR_GOAL_SLOWDOWN_DISTANCE) {
    double velocity_scaling = std::clamp(distance_to_goal / NEAR_GOAL_SLOWDOWN_DISTANCE, 0.0, 1.0);
    double unbounded_vel = approach_vel * velocity_scaling;
    if (unbounded_vel < min_approach_linear_velocity_) {
      approach_vel = min_approach_linear_velocity_;
    } else {
      approach_vel *= velocity_scaling;
    }

    // Use the lowest velocity between approach and other constraints, if all overlapping
    linear_vel = std::min(linear_vel, approach_vel);
  }

  // Speed ramp
  // https://www.wolframalpha.com/input?i=plot+1%2F%281%2Bexp%28a%2Bbx%29%29+and+a%2Bbx+where+a%3D5+and+b%3D-0.7+and+x+from+-1+to+10+and+y+from+-1+to+2
  auto now = std::chrono::system_clock::now();
  auto speed_ramp_duration = now - reset_speed_ramp_ts_;
  double t = static_cast<double>(
    std::chrono::duration_cast<std::chrono::milliseconds>(speed_ramp_duration).count()
    ) / 1.e3;
  const double const_time = 5.0;  // seconds to drive constant speed
  const double ramp_time = 10.0;  // seconds to ramp up from the constant speed to the full speed
  if (t < const_time + ramp_time) {
    double ramp = 0.05;  // constant time
    if (t > const_time) {
      // ramp
      ramp = ((1.0 - ramp) / ramp_time) * (t - const_time) + ramp;
    }

    linear_vel *= ramp;
    // RCLCPP_INFO(logger_, "t: %f, ramp: %f", t, ramp);
  }

  // Limit linear velocities to be valid
  linear_vel = std::clamp(fabs(linear_vel), 0.02, desired_linear_vel_);
  linear_vel = sign * linear_vel;

  // RCLCPP_INFO(
  //     logger_,
  //     "deviation: %.3f, curvature: %.3f, distance_to_goal: %.3f, cost_vel: %.3f, curvature_vel: %.3f, deviation_vel: %.3f, desired_linear_vel: %.3f, approach_vel: %.3f, linear_vel: %.3f",
  //     deviation_from_path, curvature, distance_to_goal, cost_vel, curvature_vel, deviation_vel, desired_linear_vel_, approach_vel, linear_vel);
}

void RegulatedPurePursuitController::setPlan(const nav_msgs::msg::Path & path)
{
  global_plan_ = path;
  resetSpeedRamp();
}

void RegulatedPurePursuitController::setSpeedLimit(
  const double & speed_limit,
  const bool & percentage)
{
  if (speed_limit == -1) {
    // magic value -1 is the nav2 controller server telling us to reset the speed ramp
    resetSpeedRamp();
  } else if (speed_limit == nav2_costmap_2d::NO_SPEED_LIMIT) {
    // Restore default value
    desired_linear_vel_ = base_desired_linear_vel_;
  } else {
    if (percentage) {
      // Speed limit is expressed in % from maximum speed of robot
      desired_linear_vel_ = base_desired_linear_vel_ * speed_limit / 100.0;
    } else {
      // Speed limit is expressed in absolute value
      desired_linear_vel_ = speed_limit;
    }
  }
}

void RegulatedPurePursuitController::resetSpeedRamp()
{
  reset_speed_ramp_ts_ = std::chrono::system_clock::now();
}

nav_msgs::msg::Path RegulatedPurePursuitController::transformGlobalPlan(
  const geometry_msgs::msg::PoseStamped & pose,
  double & distance_to_goal
)
{
  if (global_plan_.poses.empty()) {
    throw nav2_core::PlannerException("Received plan with zero length");
  }

  // let's get the pose of the robot in the frame of the plan
  geometry_msgs::msg::PoseStamped robot_pose;
  if (!transformPose(global_plan_.header.frame_id, pose, robot_pose)) {
    throw nav2_core::PlannerException("Unable to transform robot pose into global plan's frame");
  }

  // Calculate the distance from the robot to the end of the global plan
  distance_to_goal = euclidean_distance(robot_pose, global_plan_.poses.back());

  // We'll discard points on the plan that are outside the local costmap
  nav2_costmap_2d::Costmap2D * costmap = costmap_ros_->getCostmap();
  const double max_costmap_dim = std::max(costmap->getSizeInCellsX(), costmap->getSizeInCellsY());
  const double max_transform_dist = max_costmap_dim * costmap->getResolution() / 2.0;

  // First find the closest pose on the path to the robot
  auto last_pose_it = std::min(global_plan_.poses.begin() + 5, global_plan_.poses.end());
  auto transformation_begin =
    nav2_util::geometry_utils::min_by(
    global_plan_.poses.begin(), last_pose_it,
    [&robot_pose](const geometry_msgs::msg::PoseStamped & ps) {
      return euclidean_distance(robot_pose, ps);
    });

  // Find points definitely outside of the costmap so we won't transform them.
  auto transformation_end = std::find_if(
    transformation_begin + 2, end(global_plan_.poses),
    [&](const auto & global_plan_pose) {
      return euclidean_distance(robot_pose, global_plan_pose) > max_transform_dist;
    });

  // Add the next outside the costmap
  if (transformation_end < end(global_plan_.poses)) {
    transformation_end++;
  }

  // Add the previous outside the costmap
  if (transformation_begin > begin(global_plan_.poses)) {
    transformation_begin--;
  }

  // Lambda to transform a PoseStamped from global frame to local
  auto transformGlobalPoseToLocal = [&](const auto & global_plan_pose) {
      geometry_msgs::msg::PoseStamped stamped_pose, transformed_pose;
      stamped_pose.header.frame_id = global_plan_.header.frame_id;
      stamped_pose.header.stamp = robot_pose.header.stamp;
      stamped_pose.pose = global_plan_pose.pose;
      transformPose(costmap_ros_->getBaseFrameID(), stamped_pose, transformed_pose);
      return transformed_pose;
    };

  // Transform the near part of the global plan into the robot's frame of reference.
  nav_msgs::msg::Path transformed_plan;
  std::transform(
    transformation_begin, transformation_end,
    std::back_inserter(transformed_plan.poses),
    transformGlobalPoseToLocal);
  transformed_plan.header.frame_id = costmap_ros_->getBaseFrameID();
  transformed_plan.header.stamp = robot_pose.header.stamp;

  // Remove the portion of the global plan that we've already passed so we don't
  // process it on the next iteration (this is called path pruning)
  global_plan_.poses.erase(begin(global_plan_.poses), transformation_begin);
  global_path_pub_->publish(transformed_plan);

  if (transformed_plan.poses.empty()) {
    throw nav2_core::PlannerException("Resulting plan has 0 poses in it.");
  }

  return transformed_plan;
}

double RegulatedPurePursuitController::findDirectionChange(
  const geometry_msgs::msg::PoseStamped & pose)
{
  // Iterating through the global path to determine the position of the cusp
  for (unsigned int pose_id = 1; pose_id < global_plan_.poses.size() - 1; ++pose_id) {
    // We have two vectors for the dot product OA and AB. Determining the vectors.
    double oa_x = global_plan_.poses[pose_id].pose.position.x -
      global_plan_.poses[pose_id - 1].pose.position.x;
    double oa_y = global_plan_.poses[pose_id].pose.position.y -
      global_plan_.poses[pose_id - 1].pose.position.y;
    double ab_x = global_plan_.poses[pose_id + 1].pose.position.x -
      global_plan_.poses[pose_id].pose.position.x;
    double ab_y = global_plan_.poses[pose_id + 1].pose.position.y -
      global_plan_.poses[pose_id].pose.position.y;

    /* Checking for the existance of cusp, in the path, using the dot product
    and determine it's distance from the robot. If there is no cusp in the path,
    then just determine the distance to the goal location. */
    if ( (oa_x * ab_x) + (oa_y * ab_y) < 0.0) {
      auto x = global_plan_.poses[pose_id].pose.position.x - pose.pose.position.x;
      auto y = global_plan_.poses[pose_id].pose.position.y - pose.pose.position.y;
      return hypot(x, y);  // returning the distance if there is a cusp
    }
  }

  return std::numeric_limits<double>::max();
}

bool RegulatedPurePursuitController::transformPose(
  const std::string frame,
  const geometry_msgs::msg::PoseStamped & in_pose,
  geometry_msgs::msg::PoseStamped & out_pose) const
{
  if (in_pose.header.frame_id == frame) {
    out_pose = in_pose;
    return true;
  }

  try {
    tf_->transform(in_pose, out_pose, frame, transform_tolerance_);
    out_pose.header.frame_id = frame;
    return true;
  } catch (tf2::TransformException & ex) {
    RCLCPP_ERROR(logger_, "Exception in transformPose: %s", ex.what());
  }
  return false;
}
}  // namespace nav2_regulated_pure_pursuit_controller

// Register this controller as a nav2_core plugin
PLUGINLIB_EXPORT_CLASS(
  nav2_regulated_pure_pursuit_controller::RegulatedPurePursuitController,
  nav2_core::Controller)
