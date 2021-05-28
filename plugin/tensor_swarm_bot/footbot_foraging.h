#ifndef FOOTBOT_FORAGING_H
#define FOOTBOT_FORAGING_H

/*
 * Include some necessary headers.
 */
/* Definition of the CCI_Controller class. */
#include <argos3/core/control_interface/ci_controller.h>
/* Definition of the Positioning Sensor class. */
#include <argos3/plugins/robots/generic/control_interface/ci_positioning_sensor.h>
/* Definition of the differential steering actuator */
#include <argos3/plugins/robots/generic/control_interface/ci_differential_steering_actuator.h>
/* Definition of the LEDs actuator */
#include <argos3/plugins/robots/generic/control_interface/ci_leds_actuator.h>
/* Definition of the range and bearing actuator */
#include <argos3/plugins/robots/generic/control_interface/ci_range_and_bearing_actuator.h>
/* Definition of the range and bearing sensor */
#include <argos3/plugins/robots/generic/control_interface/ci_range_and_bearing_sensor.h>
/* Definition of the foot-bot proximity sensor */
#include <argos3/plugins/robots/foot-bot/control_interface/ci_footbot_proximity_sensor.h>
/* Definition of the foot-bot light sensor */
#include <argos3/plugins/robots/foot-bot/control_interface/ci_footbot_light_sensor.h>
/* Definition of the foot-bot motor ground sensor */
#include <argos3/plugins/robots/foot-bot/control_interface/ci_footbot_motor_ground_sensor.h>

/* Definitions for random number generation */
#include <argos3/core/utility/math/rng.h>

#include "ma_foraging/AIService.h"

#include <string>
#include <unordered_map>
/*
 * All the ARGoS stuff in the 'argos' namespace.
 * With this statement, you save typing argos:: every time.
 */
using namespace argos;

//#define DEBUG_LOG             1

//#define USE_FOOD_NEST_BEACON  1
//#define PERSISTENCE_CNT       20

//#define STAY_REWARD           -0.01
//#define SWITCH_REWARD         -0.5
//#define PICKUP_FOOD_REWARD    1
//#define RETURN_FOOD_REWARD    10

//#define PROBABILITY         0.3
#define ANGLE               45
#define STEP_CNT            4
#define DIE_DOWN_RESET      10
#define PURSUE_RESET        5
#define MAX_HOP_CNT         15
#define UNDEFINED_TARGET    99
#define COLLISION_DISTANCE  30
#define MIN_SAFE_DISTANCE   20
#define MAX_DISTANCE        1000

/*
 * A controller is simply an implementation of the CCI_Controller class.
 */
class CFootBotForaging : public CCI_Controller {
public:
    /*
     * This structure holds data about foraging parameters
     */
    struct SForagingParams {
        Real    m_fFoodSquareRadius;        // the radius of food/nest regions
        Real    m_fFoodPosX;                // the center x position of food region
        Real    m_fFoodPosY;                // the center y position of food region
        Real    m_fNestPosX;                // the center x position of nest region
        Real    m_fNestPosY;                // the center y position of nest region
        Real    STAY_REWARD;                // reward if robot keeps current role
        Real    SWITCH_REWARD;              // reward if robot change role
        Real    PICKUP_FOOD_REWARD;         // reward for picking up the food
        Real    RETURN_FOOD_REWARD;         // reward for returning food to nest
        Real    BEACON_REWARD;              // reward if a walker become a beacons if no beacons around
        Real    CROWDED_REWARD;             // reward if too much beacons around current robot 
        Real    SPARSE_REWARD;              // reward if very less beacons around current robot
        Real    INF_CARDINALITY_REWARD;     // reward if either food/nest cardinality is infinite
        Real    ZERO_CARDINALITY_REWARD;    // reward if either food/nest cardinality is 0 
        Real    ZERO_CARDINALITY_METHOD;    // 0: (5-cardinality)/5*ZERO_CARDINALITY_REWARD, 1: inverse cardinality/5*ZERO_CARDINALITY_REWARD
        UInt32  PERSISTENCE_CNT;            // the number of steps per decision
        Real    PROBABILITY;                // the probability to change the robot from beacon to walker state
        Real    MaxBeaconDistance;          // the max distance to count number of heard beacons
        Real    MaxDistanceDetectFoodNest;  // the max distance that can detect nearby FOOD/NEST
        Real    TurnToTargetRate;           // the rate between (0,1) the robot will turn to target if exists
        Real    RewardSplitRatio;           // the split ratio of reward between beacon vs walker

        SForagingParams();
        /* Parses the XML section for diffusion */
        void Init(TConfigurationNode& t_tree);
        /* override operator=*/
        SForagingParams& operator=(const SForagingParams &params);
    };

    /*
     * The following variables are used as parameters for the
     * diffusion algorithm. You can set their value in the <parameters>
     * section of the XML configuration file, under the
     * <controllers><footbot_foraging_controller><parameters><diffusion>
     * section.
     */
    struct SDiffusionParams {
      /*
       * Maximum tolerance for the proximity reading between
       * the robot and the closest obstacle.
       * The proximity reading is 0 when nothing is detected
       * and grows exponentially to 1 when the obstacle is
       * touching the robot.
       */
      Real Delta;
      /* Angle tolerance range to go straight. */
      CRange<CRadians> GoStraightAngleRange;

      /* Constructor */
      SDiffusionParams();

      /* Parses the XML section for diffusion */
      void Init(TConfigurationNode& t_tree);
    };

    /*
     * The following variables are used as parameters for
     * turning during navigation. You can set their value
     * in the <parameters> section of the XML configuration
     * file, under the
     * <controllers><footbot_foraging_controller><parameters><wheel_turning>
     * section.
     */
    struct SWheelTurningParams {
      /*
       * The turning mechanism.
       * The robot can be in three different turning states.
       */
      enum ETurningMechanism {
         NO_TURN = 0, // go straight
         SOFT_TURN,   // both wheels are turning forwards, but at different speeds
         HARD_TURN    // wheels are turning with opposite speeds
      } TurningMechanism;
      /*
       * Angular thresholds to change turning state.
       */
      CRadians HardTurnOnAngleThreshold;
      CRadians SoftTurnOnAngleThreshold;
      CRadians NoTurnAngleThreshold;
      /* Maximum wheel speed */
      Real MaxSpeed;
      
      void Init(TConfigurationNode& t_tree);
    };

    /*
      * Contains all the state information about the controller.
      */
    struct SStateData {
      int id;
      /* The two possible goals in which the controller can target */
      enum EGoal {
         GOAL_FOOD = 0,
         GOAL_NEST
      } Goal;
      /* The three possible states in which the controller can be */
      enum EState {
         STATE_WALKER= 0,
         STATE_BEACON,
         STATE_FOOD,
         STATE_NEST
      } State;
      /* The three possible actions in which the controller can take */
      enum EAction {
         ACTION_AVOID = 0,
         ACTION_EXPLORE,
         ACTION_PURSUE
      } Action;
      /* The three possible movement in which the controller can take */
      enum EMovement {
         MOVE_STATIONARY = 0,
         MOVE_FORWARD,
         MOVE_TURN
      } Movement;
      
      UInt32 loopCounter;
      UInt32 stepCounter;
      UInt32 pursueStepCounter;
      UInt32 currentpursueAction;
      UInt32 walkerPersistenceCnt;
      UInt32 beaconPersistenceCnt;
      UInt32 myNestHop;
      UInt32 myFoodHop;
      UInt32 target_id;
      
      /* True when the robot is in the nest or having food*/
      bool InNest;
      bool HasFood;
      float foodDist, nestDist;                 //20200228
      bool PickUpEvent, ReturnFoodEvent;        //20200228
      UInt32 PickUpBeaconId, ReturnBeaconId;    //20200228
      
      Real reward;
      
      bool turnDirectionLeft;
      UInt32 prevDecision;
      UInt32 curDecision;
      UInt32 nBeacons;
      UInt32 nWalkers;                          //20200229
      UInt32 nFoodReturnToNest;
      
      SStateData();
      void Init(TConfigurationNode& t_node);
      void Reset();
    };

    /*
      * Define the message passing between robots through range and bearing actuators
      */
    struct SMessage {
       UInt8  id;
       UInt8  state;
       UInt32 fHop;
       UInt32 nHop;
       float  dist;
       float  angle;
       float  prevDist;
       UInt32 die;

       SMessage();
    };
   
public:
    /* Class constructor. */
    CFootBotForaging();
    /* Class destructor. */
    virtual ~CFootBotForaging() {}

    /*
    * This function initializes the controller.
    * The 't_node' variable points to the <parameters> section in the XML
    * file in the <controllers><footbot_foraging_controller> section.
    */
    virtual void Init(TConfigurationNode& t_node);

    /*
    * This function is called once every time step.
    * The length of the time step is set in the XML file.
    */
    virtual void ControlStep();

    /*
    * This function resets the controller to its state right after the
    * Init().
    * It is called when you press the reset button in the GUI.
    */
    virtual void Reset();

    /*
     * Called to cleanup what done by Init() when the experiment finishes.
     * In this example controller there is no need for clean anything up,
     * so the function could have been omitted. It's here just for
     * completeness.
    */
    virtual void Destroy() {}

    /*
     * Get / Set the foraging params
     */
    inline SForagingParams& GetForagingParams() {
      return m_sForagingParams;
    }
    
    inline void SetForagingParams(SForagingParams params) {
      m_sForagingParams = params;
    }
   
    /*
     * Returns the state data
     */
    inline SStateData& GetStateData() {
      return m_sStateData;
    }
    
    /*
     * Returns the message table
     */
    inline std::unordered_map<UInt8, SMessage*>& GetMessageTable() {
      return m_sMessageTable;
    }

    /*
     * Returns the state data
     */
    inline bool isFinishUpdateState() {
        return m_bUpdateStateCalled;
    }
    
    /*
     * Manually set the position of FOOD and NEST on simulation env
     */
    inline void setFoodNestPos(CVector2 cFoodPos, CVector2 cNestPos) {
        m_cFoodPos = cFoodPos;
        m_cNestPos = cNestPos;
    }
   
    /*
     * Set new decision of role changing between BEACON and WALKER
     */
    void SetNewAction(UInt32 newDecision, bool bNewDecision);

    /*
     * Return the current observation
     */
    ma_foraging::Observation getObservation();

private:
   /*
    * Updates the state information.
    * In pratice, it sets the SStateData::InNest flag.
    * Future, more complex implementations should add their
    * state update code here.
    */
   void UpdateState(bool bNewDecision);

   /*
    * Calculates the vector to the light. Used to perform
    * phototaxis and antiphototaxis.
    */
   CVector2 CalculateVectorToLight();

   /*
    * Calculates the diffusion vector. If there is a close obstacle,
    * it points away from it; it there is none, it points forwards.
    * The b_collision parameter is used to return true or false whether
    * a collision avoidance just happened or not. It is necessary for the
    * collision rule.
    */
   CVector2 DiffusionVector(bool& b_collision);

   /*
    * Gets a direction vector as input and transforms it into wheel
    * actuation.
    */
   void SetWheelSpeedsFromVector(const CVector2& c_heading);

   /*
    * Executes the resting state.
    */
   //void Beacon();

   /*
    * Executes the exploring state.
    */
   //void Walker();

   /*
    * Executes foraging algorithms
    */
   void nestSearch();
   void foodSearch();
   
   bool tryToPickUpFood();
   bool tryToDropOffFood();
   
   void receiveMessages();
   void sendMessages();
   
   size_t determineHopcount();
   bool isItBeacon(size_t id);
   size_t howManyBeacons();
   bool shouldIExplore();
   bool isObstacle();
   
   bool acquireTarget(size_t type);
   void pursue();
   void runMotors();
   void turnOnLights();

private:
   /* Pointer to the differential steering actuator */
   CCI_DifferentialSteeringActuator* m_pcWheels;
   /* Pointer to the LEDs actuator */
   CCI_LEDsActuator* m_pcLEDs;
   /* Pointer to the range and bearing actuator */
   CCI_RangeAndBearingActuator* m_pcRABA;
   /* Pointer to the range and bearing sensor */
   CCI_RangeAndBearingSensor* m_pcRABS;
   /* Pointer to the foot-bot proximity sensor */
   CCI_FootBotProximitySensor* m_pcProximity;
   /* Pointer to the foot-bot light sensor */
   CCI_FootBotLightSensor* m_pcLight;
   /* Pointer to the foot-bot motor ground sensor */
   CCI_FootBotMotorGroundSensor* m_pcGround;
   /* Pointer to the positioning sensor */
   CCI_PositioningSensor* m_pcPosSensor;
   /* The random number generator */
   CRandom::CRNG* m_pcRNG;
   /* The foraging parameters */
   SForagingParams m_sForagingParams;
   /* The controller state information */
   SStateData m_sStateData;
   /* The message receives from nearby robots */
   std::unordered_map<UInt8, SMessage*> m_sMessageTable;
   /* The turning parameters */
   SWheelTurningParams m_sWheelTurningParams;
   /* The diffusion parameters */
   SDiffusionParams m_sDiffusionParams;
   /* The flag to indicate update robot after receiving new decision or not */
   bool m_bUpdateStateCalled;
   /* The position of FOOD and NEST on the simulation enviroment*/
   CVector2 m_cFoodPos, m_cNestPos;
};

#endif
