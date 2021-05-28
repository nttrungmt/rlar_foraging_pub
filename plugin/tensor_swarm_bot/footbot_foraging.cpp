/* Include the controller definition */
#include "footbot_foraging.h"
/* Function definitions for XML parsing */
#include <argos3/core/utility/configuration/argos_configuration.h>
/* 2D vector definition */
#include <argos3/core/utility/math/vector2.h>
/* Logging */
#include <argos3/core/utility/logging/argos_log.h>
#include <sstream>      // std::stringstream

using namespace ma_foraging;

/****************************************/
/****************************************/
CFootBotForaging::SForagingParams::SForagingParams() :
    m_fFoodSquareRadius(0.6),
    m_fFoodPosX(0.0), m_fFoodPosY(2.6), m_fNestPosX(0.0), m_fNestPosY(-2.6),
    STAY_REWARD(-0.05),
    SWITCH_REWARD(-0.5),
    PICKUP_FOOD_REWARD(1),
    RETURN_FOOD_REWARD(10),
    BEACON_REWARD(0.0),
    CROWDED_REWARD(0.0),
    SPARSE_REWARD(0.0),
    INF_CARDINALITY_REWARD(0.0),
    ZERO_CARDINALITY_REWARD(0.0),
    ZERO_CARDINALITY_METHOD(0.0),
    PERSISTENCE_CNT(20),
    PROBABILITY(0.3),
    MaxBeaconDistance(150),
    MaxDistanceDetectFoodNest(0.9),
    TurnToTargetRate(0.25),
    RewardSplitRatio(0.5)        {}

void CFootBotForaging::SForagingParams::Init(TConfigurationNode& t_node) {
    try {
        GetNodeAttribute(t_node, "radius", m_fFoodSquareRadius);
        GetNodeAttribute(t_node, "foodPosX", m_fFoodPosX);
        GetNodeAttribute(t_node, "foodPosY", m_fFoodPosY);
        GetNodeAttribute(t_node, "nestPosX", m_fNestPosX);
        GetNodeAttribute(t_node, "nestPosY", m_fNestPosY);
        ////////////////////////////////////////////////////////////
        GetNodeAttribute(t_node, "stay_reward", STAY_REWARD);
        GetNodeAttribute(t_node, "switch_reward", SWITCH_REWARD);
        GetNodeAttribute(t_node, "pickup_food_reward", PICKUP_FOOD_REWARD);
        GetNodeAttribute(t_node, "return_food_reward", RETURN_FOOD_REWARD);
        ///////////////////////////////////////////////////////
        GetNodeAttribute(t_node, "crowded_reward", CROWDED_REWARD);
        GetNodeAttribute(t_node, "beacon_reward", BEACON_REWARD);
        GetNodeAttribute(t_node, "sparse_reward", SPARSE_REWARD);
        GetNodeAttribute(t_node, "inf_cardinality_reward", INF_CARDINALITY_REWARD);
        GetNodeAttribute(t_node, "zero_cardinality_reward", ZERO_CARDINALITY_REWARD);
        if(NodeAttributeExists(t_node,"zero_cardinality_method"))
            GetNodeAttribute(t_node, "zero_cardinality_method", ZERO_CARDINALITY_METHOD);
        ///////////////////////////////////////////////////////
        GetNodeAttribute(t_node, "persistence_cnt", PERSISTENCE_CNT);
        GetNodeAttribute(t_node, "beacon_to_walker_prob", PROBABILITY);
        GetNodeAttribute(t_node, "max_beacon_distance", MaxBeaconDistance);
        GetNodeAttribute(t_node, "max_distance_detect_food_nest", MaxDistanceDetectFoodNest);
        GetNodeAttribute(t_node, "turn_to_target_rate", TurnToTargetRate);
        GetNodeAttribute(t_node, "reward_split_ratio", RewardSplitRatio);
    }
    catch(CARGoSException& ex) {
        THROW_ARGOSEXCEPTION_NESTED("Error initializing controller foraging parameters.", ex);
    }
}

CFootBotForaging::SForagingParams& CFootBotForaging::SForagingParams::operator=(const CFootBotForaging::SForagingParams &params) {
    m_fFoodSquareRadius = params.m_fFoodSquareRadius;
    m_fFoodPosX = params.m_fFoodPosX;
    m_fFoodPosY = params.m_fFoodPosY;
    m_fNestPosX = params.m_fNestPosX;
    m_fNestPosY = params.m_fNestPosY;
    //////////////////////////////////////////////////////////////////
    STAY_REWARD = params.STAY_REWARD;
    SWITCH_REWARD = params.SWITCH_REWARD;
    PICKUP_FOOD_REWARD = params.PICKUP_FOOD_REWARD;
    RETURN_FOOD_REWARD = params.RETURN_FOOD_REWARD;
    //////////////////////////////////////////////////////////
    BEACON_REWARD = params.BEACON_REWARD;
    CROWDED_REWARD = params.CROWDED_REWARD;
    SPARSE_REWARD = params.SPARSE_REWARD;
    INF_CARDINALITY_REWARD = params.INF_CARDINALITY_REWARD;
    ZERO_CARDINALITY_REWARD = params.ZERO_CARDINALITY_REWARD;
    ZERO_CARDINALITY_METHOD = params.ZERO_CARDINALITY_METHOD;
    //////////////////////////////////////////////////////////
    PERSISTENCE_CNT = params.PERSISTENCE_CNT;
    PROBABILITY = params.PROBABILITY;
    MaxBeaconDistance = params.MaxBeaconDistance;
    MaxDistanceDetectFoodNest = params.MaxDistanceDetectFoodNest;
    TurnToTargetRate = params.TurnToTargetRate;
    RewardSplitRatio = params.RewardSplitRatio;
}

/****************************************/
/****************************************/
CFootBotForaging::SDiffusionParams::SDiffusionParams() :
    GoStraightAngleRange(CRadians(-1.0f), CRadians(1.0f)) {}

void CFootBotForaging::SDiffusionParams::Init(TConfigurationNode& t_node) {
    try {
        CRange<CDegrees> cGoStraightAngleRangeDegrees(CDegrees(-10.0f), CDegrees(10.0f));
        GetNodeAttribute(t_node, "go_straight_angle_range", cGoStraightAngleRangeDegrees);
        GoStraightAngleRange.Set(ToRadians(cGoStraightAngleRangeDegrees.GetMin()),
                               ToRadians(cGoStraightAngleRangeDegrees.GetMax()));
        GetNodeAttribute(t_node, "delta", Delta);
    }
    catch(CARGoSException& ex) {
        THROW_ARGOSEXCEPTION_NESTED("Error initializing controller diffusion parameters.", ex);
    }
}

/****************************************/
/****************************************/
void CFootBotForaging::SWheelTurningParams::Init(TConfigurationNode& t_node) {
    try {
        TurningMechanism = NO_TURN;
        CDegrees cAngle;
        GetNodeAttribute(t_node, "hard_turn_angle_threshold", cAngle);
        HardTurnOnAngleThreshold = ToRadians(cAngle);
        GetNodeAttribute(t_node, "soft_turn_angle_threshold", cAngle);
        SoftTurnOnAngleThreshold = ToRadians(cAngle);
        GetNodeAttribute(t_node, "no_turn_angle_threshold", cAngle);
        NoTurnAngleThreshold = ToRadians(cAngle);
        GetNodeAttribute(t_node, "max_speed", MaxSpeed);
    }
    catch(CARGoSException& ex) {
        THROW_ARGOSEXCEPTION_NESTED("Error initializing controller wheel turning parameters.", ex);
    }
}

/****************************************/
/****************************************/
CFootBotForaging::SStateData::SStateData() {}

void CFootBotForaging::SStateData::Init(TConfigurationNode& t_node) {}

void CFootBotForaging::SStateData::Reset() {
    Goal = GOAL_FOOD;
    State = STATE_WALKER;
    Action = ACTION_EXPLORE;
    Movement = MOVE_FORWARD;
    loopCounter = 0;
    stepCounter = 0;
    pursueStepCounter = 0;
    currentpursueAction = SStateData::MOVE_FORWARD;
    walkerPersistenceCnt = 0;
    beaconPersistenceCnt = 0;
    myNestHop = MAX_HOP_CNT;
    myFoodHop = MAX_HOP_CNT;
    target_id = UNDEFINED_TARGET;
    
    InNest = false;
    HasFood = false;
    //++++++++ 20200228 ++++++++
    foodDist = -1.0;
    nestDist = -1.0;
    PickUpEvent = false;
    ReturnFoodEvent = false;
    PickUpBeaconId = UNDEFINED_TARGET;
    ReturnBeaconId = UNDEFINED_TARGET;
    //--------------------------
    turnDirectionLeft = true;
    prevDecision = 0;
    curDecision = 0;
    nBeacons = 0;
    nWalkers = 0;
    reward = 0;
    nFoodReturnToNest = 0;
}

/****************************************/
/****************************************/
CFootBotForaging::SMessage::SMessage(): 
    id(0), state(0), 
    fHop(MAX_HOP_CNT), nHop(MAX_HOP_CNT), 
    dist(0.0f),angle(0.0f) {}

/****************************************/
/****************************************/
CFootBotForaging::CFootBotForaging() :
    m_pcWheels(NULL),
    m_pcLEDs(NULL),
    m_pcRABA(NULL),
    m_pcRABS(NULL),
    m_pcProximity(NULL),
    m_pcLight(NULL),
    m_pcGround(NULL),
    m_pcPosSensor(NULL),
    m_pcRNG(NULL) {}

/****************************************/
/****************************************/
void CFootBotForaging::Init(TConfigurationNode& t_node) {
    try {
        /* Initialize sensors/actuators */
        m_pcWheels    = GetActuator<CCI_DifferentialSteeringActuator>("differential_steering");
        m_pcLEDs      = GetActuator<CCI_LEDsActuator                >("leds"                 );
        m_pcRABA      = GetActuator<CCI_RangeAndBearingActuator     >("range_and_bearing"    );
        m_pcRABS      = GetSensor  <CCI_RangeAndBearingSensor       >("range_and_bearing"    );
        m_pcProximity = GetSensor  <CCI_FootBotProximitySensor      >("footbot_proximity"    );
        m_pcLight     = GetSensor  <CCI_FootBotLightSensor          >("footbot_light"        );
        m_pcGround    = GetSensor  <CCI_FootBotMotorGroundSensor    >("footbot_motor_ground" );
        m_pcPosSensor = GetSensor  <CCI_PositioningSensor           >("positioning" );
        /* Parse foraging parameters */
        //m_sForagingParams.Init(GetNode(t_node, "foraging"));
        /* Diffusion algorithm */
        m_sDiffusionParams.Init(GetNode(t_node, "diffusion"));
        /* Wheel turning */
        m_sWheelTurningParams.Init(GetNode(t_node, "wheel_turning"));
        /* Controller state */
        //m_sStateData.Init(GetNode(t_node, "state"));
        m_sStateData.id = atoi(GetId().substr(2).c_str());
    }
    catch(CARGoSException& ex) {
        THROW_ARGOSEXCEPTION_NESTED("Error initializing the foot-bot foraging controller for robot \"" << GetId() << "\"", ex);
    }
    /* Initialize other stuff */
    /* Create a random number generator. We use the 'argos' category so
       that creation, reset, seeding and cleanup are managed by ARGoS. */
    m_pcRNG = CRandom::CreateRNG("argos");
    /* Reset all the states */
    Reset();
}

/****************************************/
/****************************************/
void CFootBotForaging::Reset() {
    m_bUpdateStateCalled = false;
    /* Reset robot state */
    m_sStateData.Reset();
    /* Set LED color */
    turnOnLights();
    /* Clear up the last communication result */
    m_pcRABA->ClearData();
    CByteArray ba;
    ba << (UInt8)m_sStateData.id;
    ba << (UInt8)m_sStateData.State;
    ba << (int)MAX_HOP_CNT;
    ba << (int)MAX_HOP_CNT;
    //std::cout << GetId() << " - Size of input in Reset: " << ba.Size() << std::endl;
    m_pcRABA->SetData(ba);
    m_bUpdateStateCalled = true;
}

/****************************************/
/****************************************/
void CFootBotForaging::ControlStep() {
    // m_sStateData.loopCounter += 1;
    // receiveMessages();
    // if(!m_sStateData.initialized) {
        // m_sStateData.curDecision = 0;
        // m_sStateData.initialized = true;
    // }
    // switch(m_sStateData.State) {
        // case SStateData::STATE_BEACON: {
            // Beacon();
            // break;
        // }
        // case SStateData::STATE_WALKER: {
            // Walker();
            // break;
        // }
        // case SStateData::STATE_FOOD:
        // case SStateData::STATE_NEST: {
            // break;
        // }
        // default: {
            // LOGERR << "We can't be here, there's a bug!" << std::endl;
        // }
    // }
    // runMotors();
    // turnOnLights();
    // sendMessages();
}

void CFootBotForaging::SetNewAction(UInt32 newDecision, bool bNewDecision) {
    m_bUpdateStateCalled = false;
    m_sStateData.prevDecision = m_sStateData.State;
    m_sStateData.curDecision = newDecision;
    UpdateState(bNewDecision);
    
    m_sStateData.loopCounter += 1;
    receiveMessages();
    m_sStateData.nBeacons = howManyBeacons();
    //++++20200228: update distance to food, nest if it can sense
    const CCI_PositioningSensor::SReading& sReading = m_pcPosSensor->GetReading();
    m_sStateData.foodDist = (CVector2(sReading.Position[0], sReading.Position[1]) - m_cFoodPos).SquareLength();
    m_sStateData.nestDist = (CVector2(sReading.Position[0], sReading.Position[1]) - m_cNestPos).SquareLength();
    //-----------------------------------------------------------
    
    switch(m_sStateData.State) {
        case SStateData::STATE_BEACON: {
            if(m_sStateData.beaconPersistenceCnt > 0) {
                m_sStateData.beaconPersistenceCnt -= 1;
            }
            break;
        }
        case SStateData::STATE_WALKER: {
            if(m_sStateData.Goal == SStateData::GOAL_NEST) {
                nestSearch();
            } else {
                foodSearch();
            }
            if(m_sStateData.walkerPersistenceCnt > 0) {
                m_sStateData.walkerPersistenceCnt -= 1;
            }
            break;
        }
        case SStateData::STATE_FOOD:
        case SStateData::STATE_NEST: {
            break;
        }
        default: {
            LOGERR << "We can't be here, there's a bug!" << std::endl;
        }
    }
    
    /////////////////////////////////////////////////
    //20200217 trung add penalty / reward for beacons at last persistence step
    if( m_sStateData.State == SStateData::STATE_BEACON && m_sStateData.beaconPersistenceCnt == 0 ) {
        //if(m_sStateData.nBeacons >= 4)
        //    m_sStateData.reward += m_sForagingParams.CROWDED_REWARD;
        if(m_sStateData.myFoodHop == MAX_HOP_CNT or m_sStateData.myNestHop == MAX_HOP_CNT)
            m_sStateData.reward += m_sForagingParams.INF_CARDINALITY_REWARD;
        //if(m_sStateData.myFoodHop == 0 or m_sStateData.myNestHop == 0)
        if(m_sForagingParams.ZERO_CARDINALITY_METHOD == 0.0) {
            if(m_sStateData.myFoodHop < 5)
                m_sStateData.reward += (5-m_sStateData.myFoodHop)/5 * m_sForagingParams.ZERO_CARDINALITY_REWARD;
            if(m_sStateData.myNestHop < 5)
                m_sStateData.reward += (5-m_sStateData.myNestHop)/5 * m_sForagingParams.ZERO_CARDINALITY_REWARD;
        } else {
            if(m_sStateData.myFoodHop < 5)
                m_sStateData.reward += m_sStateData.myFoodHop/5 * m_sForagingParams.ZERO_CARDINALITY_REWARD;
            if(m_sStateData.myNestHop < 5)
                m_sStateData.reward += m_sStateData.myNestHop/5 * m_sForagingParams.ZERO_CARDINALITY_REWARD;
        }
    } else if ( m_sStateData.State == SStateData::STATE_WALKER && m_sStateData.walkerPersistenceCnt == 0 ) {
        //if(m_sStateData.nBeacons == 0)
        //    m_sStateData.reward += m_sForagingParams.SPARSE_REWARD;
    }
    /////////////////////////////////////////////////
    
    runMotors();
    turnOnLights();
    sendMessages();
    m_bUpdateStateCalled = true;
}

ma_foraging::Observation CFootBotForaging::getObservation() {
    float max_dist = -1.0;
    float min_dist = MAX_DISTANCE;
    int   food_farthest = -1, food_closest = -1, nest_farthest = -1, nest_closest = -1;
    for( const auto& n : m_sMessageTable ) {
        if(isItBeacon(n.first)) {
            if(n.second->dist > max_dist) {
                max_dist = n.second->dist;
                food_farthest = n.second->fHop;
                nest_farthest = n.second->nHop;
            }
            if(n.second->dist < min_dist) {
                min_dist = n.second->dist;
                food_closest = n.second->fHop;
                nest_closest = n.second->nHop;
            }
        }
    }
    
    Observation obs;
    obs.curState = m_sStateData.State;
    //obs.walkerPersistenceCnt = m_sStateData.walkerPersistenceCnt;
    //obs.beaconPersistenceCnt = m_sStateData.beaconPersistenceCnt;
    obs.numBeacons = m_sStateData.nBeacons;
    obs.hasFood = m_sStateData.HasFood;
    obs.minDist = (min_dist < MAX_DISTANCE) ? min_dist : MAX_DISTANCE;
    obs.maxDist = (max_dist > -1.0) ? max_dist : MAX_DISTANCE;
    obs.foodClosest = food_closest;
    obs.nestClosest = nest_closest;
    obs.foodFarthest = food_farthest;
    obs.nestFarthest = nest_farthest;
    obs.numFoodReturn = m_sStateData.nFoodReturnToNest;
    //+++++++++++++20200228+++++++++++++++++++
    //+++++++++++++20200325: use the true foodDist and nestDist features +++++++++++++++++++
    //obs.foodDist = (m_sStateData.foodDist <= m_sForagingParams.MaxDistanceDetectFoodNest) ? m_sStateData.foodDist : MAX_DISTANCE;
    //obs.nestDist = (m_sStateData.nestDist <= m_sForagingParams.MaxDistanceDetectFoodNest) ? m_sStateData.nestDist : MAX_DISTANCE;
    obs.foodDist = m_sStateData.foodDist;
    obs.nestDist = m_sStateData.nestDist;
    //----------------------------------------
    obs.numWalkers = m_sStateData.nWalkers;
    //obs.fHop = (m_sStateData.State == SStateData::STATE_WALKER) ? MAX_HOP_CNT : m_sStateData.myFoodHop;
    //obs.nHop = (m_sStateData.State == SStateData::STATE_WALKER) ? MAX_HOP_CNT : m_sStateData.myNestHop;
    //----------------------------------------
#ifdef DEBUG_LOG
    if(m_sStateData.reward != 0.0) {
        std::stringstream ss;
        ss  << m_sStateData.loopCounter << "-" << GetId()
            << " prevState="    << m_sStateData.prevDecision            << " newState=" << m_sStateData.State
            << " hasFood="      << m_sStateData.HasFood                 << " inNest=" << m_sStateData.InNest
            << " wPsCnt="       << m_sStateData.walkerPersistenceCnt    << " bPsCnt" << m_sStateData.beaconPersistenceCnt
            << " fHop="         << m_sStateData.myFoodHop               << " nHop=" << m_sStateData.myNestHop
            << " nBeacons="     << m_sStateData.nBeacons                << " reward=" << m_sStateData.reward; 
        printf("%s\n", ss.str().c_str());
    }
#endif
    return obs;
}

/****************************************/
/****************************************/
void CFootBotForaging::UpdateState(bool bNewDecision) {
    m_sStateData.reward = 0;
    if(m_sStateData.State == SStateData::STATE_BEACON) {
        if(m_sStateData.curDecision == 0) {                         //BEACON change to WALKER
            m_sStateData.State = SStateData::STATE_WALKER;
            m_sStateData.Action = SStateData::ACTION_PURSUE;
            m_sStateData.Movement = SStateData::MOVE_FORWARD;
            m_sStateData.walkerPersistenceCnt = m_sForagingParams.PERSISTENCE_CNT;
            m_sStateData.beaconPersistenceCnt = m_sForagingParams.PERSISTENCE_CNT;
            m_sStateData.target_id = UNDEFINED_TARGET;
            if(bNewDecision) {
                m_sStateData.reward += m_sForagingParams.SWITCH_REWARD;
                if(m_sStateData.nBeacons == 0)
                    m_sStateData.reward += m_sForagingParams.SPARSE_REWARD;
            }
        } else {
            m_sStateData.Movement = SStateData::MOVE_STATIONARY;
            //20200218  add STAY_REWARD as BEACON only when making new decision
            //          add small reward if continue as beacon because no nearby beacons previously
            if(bNewDecision) {
                m_sStateData.beaconPersistenceCnt = m_sForagingParams.PERSISTENCE_CNT;
                m_sStateData.reward += m_sForagingParams.STAY_REWARD;
                if(m_sStateData.nBeacons == 0)
                    m_sStateData.reward += m_sForagingParams.BEACON_REWARD;
                if(m_sStateData.nBeacons >= 4)
                    m_sStateData.reward += m_sForagingParams.CROWDED_REWARD;
            }
        }
    } else {
        //WALKER State
        if(m_sStateData.curDecision == 1) {                         //WALKER change to BEACON
            m_sStateData.State = SStateData::STATE_BEACON;
            m_sStateData.Movement = SStateData::MOVE_STATIONARY;
            m_sStateData.beaconPersistenceCnt = m_sForagingParams.PERSISTENCE_CNT;
            m_sStateData.walkerPersistenceCnt = m_sForagingParams.PERSISTENCE_CNT;
            m_sStateData.target_id = UNDEFINED_TARGET;
            //20200218  add small reward if continue as beacon because no nearby beacons previously
            if(bNewDecision) {
                m_sStateData.reward += m_sForagingParams.SWITCH_REWARD;
                if(m_sStateData.nBeacons == 0)
                    m_sStateData.reward += m_sForagingParams.BEACON_REWARD;
                if(m_sStateData.nBeacons >= 4)
                    m_sStateData.reward += m_sForagingParams.CROWDED_REWARD;
            }
        } else {
            //m_sStateData.Movement = SStateData::MOVE_FORWARD;
            //20200218  add STAY_REWARD as WALKER only when making new decision
            if(bNewDecision) {
                m_sStateData.walkerPersistenceCnt = m_sForagingParams.PERSISTENCE_CNT;
                m_sStateData.reward += m_sForagingParams.STAY_REWARD;
                if(m_sStateData.nBeacons == 0)
                    m_sStateData.reward += m_sForagingParams.SPARSE_REWARD;
            }
        }
    }
}

/****************************************/
/****************************************/
/*void CFootBotForaging::Beacon() {
    if(m_sStateData.beaconPersistenceCnt > 0) {
        m_sStateData.beaconPersistenceCnt -= 1;
        m_sStateData.reward += m_sForagingParams.STAY_REWARD;
        return;
    }

    size_t nBeacons = howManyBeacons();
    if(nBeacons >= 3) {
        double r = ((double) rand() / (RAND_MAX));
        if(r < m_sForagingParams.PROBABILITY) {
            m_sStateData.curDecision = 0;        //change from BEACON to WALKER
            //std::cout << GetId() << " change from BEACON to WALKER state with r=" << r << std::endl;
        } else {
            m_sStateData.curDecision = 1;        //continue in BEACON state
            //std::cout << GetId() << " not change BEACON state with r=" << r << std::endl;
        }
    } else {
        //std::cout << GetId() << " not change BEACON state because less than 3 beacons:" << nBeacons << std::endl;
    }
    
    UpdateState();
}*/

/****************************************/
/****************************************/
/*void CFootBotForaging::Walker() {
    if(m_sStateData.Goal == SStateData::GOAL_NEST) {       //m_sStateData.HasFood
        nestSearch();
    } else {
        foodSearch();
    }
   
    if(m_sStateData.walkerPersistenceCnt > 0) {
        m_sStateData.walkerPersistenceCnt -= 1;
        m_sStateData.reward += m_sForagingParams.STAY_REWARD;
        return;
    }
    
    size_t nBeacons = howManyBeacons();
    if(nBeacons < 2) {
        m_sStateData.curDecision = 1;        //change to BEACON state
        //std::cout << GetId() << " change from WALKER to BEACON state, nBeacons=" << nBeacons << std::endl;
    }
    
    UpdateState();
}*/

/****************************************/
/****************************************/
void CFootBotForaging::nestSearch() {
    if(tryToDropOffFood())
        return;
    
    if(m_sStateData.Action == SStateData::ACTION_AVOID) {
        if(isObstacle()) {
            m_sStateData.Movement = SStateData::MOVE_TURN;
        } else {
            bool bFoundTarget = acquireTarget(SStateData::GOAL_NEST);
            m_sStateData.Action = (bFoundTarget) ? SStateData::ACTION_PURSUE : SStateData::ACTION_EXPLORE;
            m_sStateData.Movement = SStateData::MOVE_FORWARD;
        }
    } else if (m_sStateData.Action == SStateData::ACTION_EXPLORE) {
        if(m_sStateData.stepCounter > STEP_CNT) {
            m_sStateData.stepCounter = 0;
            if(acquireTarget(SStateData::GOAL_NEST))
                m_sStateData.Action = SStateData::ACTION_PURSUE;
            m_sStateData.Movement = SStateData::MOVE_FORWARD;
        } else {
            if(isObstacle()) {
                //m_sStateData.Action = SStateData::ACTION_AVOID;
                m_sStateData.Movement = SStateData::MOVE_TURN;
            } else {
                m_sStateData.Movement = SStateData::MOVE_FORWARD;
                m_sStateData.stepCounter += 1;
            }
        }
    } else {
        if(shouldIExplore()) {
            m_sStateData.stepCounter = 0;
            m_sStateData.Action = SStateData::ACTION_EXPLORE;
            m_sStateData.Movement = SStateData::MOVE_FORWARD;
        } else {
            if(isObstacle()) {
                //m_sStateData.Action = SStateData::ACTION_AVOID;
                m_sStateData.Movement = SStateData::MOVE_TURN;
            } else {
                if( acquireTarget(SStateData::GOAL_NEST) ) {
                    pursue();
                } else {
                    m_sStateData.Action = SStateData::ACTION_EXPLORE;
                    m_sStateData.Movement = SStateData::MOVE_FORWARD;
                }
            }
        }
    }
}

void CFootBotForaging::foodSearch() {
    if(tryToPickUpFood())
        return;
    
    if(m_sStateData.Action == SStateData::ACTION_AVOID) {
        if(isObstacle()) {
            m_sStateData.Movement = SStateData::MOVE_TURN;
        } else {
            bool bFoundTarget = acquireTarget(SStateData::GOAL_FOOD);
            m_sStateData.Action = (bFoundTarget) ? SStateData::ACTION_PURSUE : SStateData::ACTION_EXPLORE;
            m_sStateData.Movement = SStateData::MOVE_FORWARD;
        }
    } else if(m_sStateData.Action == SStateData::ACTION_EXPLORE) {
        if(m_sStateData.stepCounter > STEP_CNT) {
            m_sStateData.stepCounter = 0;
            if(acquireTarget(SStateData::GOAL_FOOD))
                m_sStateData.Action = SStateData::ACTION_PURSUE;
            m_sStateData.Movement = SStateData::MOVE_FORWARD;
        } else {
            if(isObstacle()) {
                //m_sStateData.Action = SStateData::ACTION_AVOID;
                m_sStateData.Movement = SStateData::MOVE_TURN;
            } else {
                m_sStateData.Movement = SStateData::MOVE_FORWARD;
                m_sStateData.stepCounter += 1;
            }
        }
    } else {
        if(shouldIExplore()) {
            m_sStateData.stepCounter = 0;
            m_sStateData.Action = SStateData::ACTION_EXPLORE;
            m_sStateData.Movement = SStateData::MOVE_FORWARD;
        } else {
            if(isObstacle()) {
                //m_sStateData.Action = SStateData::ACTION_AVOID;
                m_sStateData.Movement = SStateData::MOVE_TURN;
            } else {
                if( acquireTarget(SStateData::GOAL_FOOD) ) {
                    pursue();
                } else {
                    m_sStateData.Action = SStateData::ACTION_EXPLORE;
                    m_sStateData.Movement = SStateData::MOVE_FORWARD;
                }
            }
        }
    }
}

bool CFootBotForaging::tryToPickUpFood() {
    //UpdateState();
    //const CCI_PositioningSensor::SReading& sReading = m_pcPosSensor->GetReading();
    //Real rFoodDist = (CVector2(sReading.Position[0], sReading.Position[1]) - m_cFoodPos).SquareLength();
    /* Read stuff from the ground sensor */
    const CCI_FootBotMotorGroundSensor::TReadings& tGroundReads = m_pcGround->GetReadings();
    if(/*tGroundReads[2].Value == 0.0f && tGroundReads[3].Value == 0.0f &&*/
       (tGroundReads[0].Value == 0.0f && tGroundReads[1].Value == 0.0f)
        || m_sStateData.foodDist <= m_sForagingParams.MaxDistanceDetectFoodNest) {
       m_sStateData.HasFood = true;
       m_sStateData.Goal = SStateData::GOAL_NEST;
       m_sStateData.InNest = false;
#ifdef USE_FOOD_NEST_BEACON
       //try to determine if there is one robot that server as FOOD BEACON (FIXED)
       bool bFound = false;
       for( const auto& n : m_sMessageTable ) {
          if(n.second->state == SStateData::STATE_FOOD) {
             bFound = true;
             break;
          }
       }
       if(!bFound) {
          //there is no FOOD BEACON => transform this robot into such FOOD BEACON
          m_sStateData.State = SStateData::STATE_FOOD;
          m_sStateData.Movement = SStateData::MOVE_STATIONARY;
          m_sStateData.myFoodHop = 0;
          std::cout << GetId() << " become FOOD BEACON!" << std::endl;
          return true;
       }
#endif
    } else if( (tGroundReads[0].Value > 0.25f && tGroundReads[0].Value < 0.75f &&
                tGroundReads[1].Value > 0.25f && tGroundReads[1].Value < 0.75f
                /*&& tGroundReads[2].Value > 0.25f && tGroundReads[2].Value < 0.75f &&
                     tGroundReads[3].Value > 0.25f && tGroundReads[3].Value < 0.75f*/) ) {
#ifdef USE_FOOD_NEST_BEACON
       //try to determine if there is one robot that server as NEST BEACON (FIXED)
       bool bFound = false;
       for( const auto& n : m_sMessageTable ) {
          if(n.second->state == SStateData::STATE_NEST) {
             bFound = true;
             break;
          }
       }
       if(!bFound) {
          //there is no NEST BEACON => transform this robot into such NEST BEACON
          m_sStateData.InNest = true;
          m_sStateData.Goal = SStateData::GOAL_FOOD;
          m_sStateData.HasFood = false;
          m_sStateData.State = SStateData::STATE_NEST;
          m_sStateData.Movement = SStateData::MOVE_STATIONARY;
          m_sStateData.myNestHop = 0;
          std::cout << GetId() << " become NEST BEACON!" << std::endl;
          return true;
       }
#endif
    }
    
    if(m_sStateData.HasFood) {
        //if(m_sStateData.Action == SStateData::ACTION_PURSUE) {
        if(m_sStateData.target_id != UNDEFINED_TARGET) {
            std::cout << GetId() << " picked up food at " << m_sStateData.loopCounter << " FULL REWARD" << std::endl;
            m_sStateData.reward += m_sForagingParams.PICKUP_FOOD_REWARD*(1-m_sForagingParams.RewardSplitRatio);     //20200303
            m_sStateData.PickUpEvent = true;                        //20200228
            m_sStateData.PickUpBeaconId = m_sStateData.target_id;   //20200228
        } else {
            std::cout << GetId() << " picked up food at " << m_sStateData.loopCounter << " NO REWARD" << std::endl;
            //m_sStateData.reward += m_sForagingParams.PICKUP_FOOD_REWARD/4;
        }
        m_sStateData.Action = SStateData::ACTION_EXPLORE;
        return true;
    } else {
        return false;
    }
}

bool CFootBotForaging::tryToDropOffFood() {
    //UpdateState();
    //const CCI_PositioningSensor::SReading& sReading = m_pcPosSensor->GetReading();
    //Real rNestDist = (CVector2(sReading.Position[0], sReading.Position[1]) - m_cNestPos).SquareLength();
    /* Read stuff from the ground sensor */
    const CCI_FootBotMotorGroundSensor::TReadings& tGroundReads = m_pcGround->GetReadings();
    if( (tGroundReads[0].Value > 0.25f && tGroundReads[0].Value < 0.75f &&
         tGroundReads[1].Value > 0.25f && tGroundReads[1].Value < 0.75f 
            /*&& tGroundReads[2].Value > 0.25f && tGroundReads[2].Value < 0.75f &&
               tGroundReads[3].Value > 0.25f && tGroundReads[3].Value < 0.75f*/)
        || m_sStateData.nestDist <= m_sForagingParams.MaxDistanceDetectFoodNest ) {
       m_sStateData.InNest = true;
       m_sStateData.Goal = SStateData::GOAL_FOOD;
       m_sStateData.HasFood = false;
#ifdef USE_FOOD_NEST_BEACON
       //try to determine if there is one robot that server as NEST BEACON (FIXED)
       bool bFound = false;
       for( const auto& n : m_sMessageTable ) {
          if(n.second->state == SStateData::STATE_NEST) {
             bFound = true;
             break;
          }
       }
       if(!bFound) {
          //there is no NEST BEACON => transform this robot into such NEST BEACON
          m_sStateData.State = SStateData::STATE_NEST;
          m_sStateData.Movement = SStateData::MOVE_STATIONARY;
          m_sStateData.myNestHop = 0;
          std::cout << GetId() << " become NEST BEACON!" << std::endl;
       }
#endif
    }

    if(m_sStateData.InNest) {
        //if(m_sStateData.Action == SStateData::ACTION_PURSUE) {
        if(m_sStateData.target_id != UNDEFINED_TARGET) {
            std::cout << GetId() << " dropped off food at " << m_sStateData.loopCounter << " FULL REWARD" << std::endl;
            m_sStateData.reward += m_sForagingParams.RETURN_FOOD_REWARD*(1-m_sForagingParams.RewardSplitRatio);     //20200303
            m_sStateData.ReturnFoodEvent = true;                    //20200228
            m_sStateData.ReturnBeaconId = m_sStateData.target_id;   //20200228
        } else {
            std::cout << GetId() << " dropped off food at " << m_sStateData.loopCounter << " NO REWARD" << std::endl;
            //m_sStateData.reward += m_sForagingParams.RETURN_FOOD_REWARD/20;
        }
        m_sStateData.nFoodReturnToNest++;
        m_sStateData.Action = SStateData::ACTION_EXPLORE;
        return true;
    } else {
        return false;
    }
}

void CFootBotForaging::receiveMessages() {
    const CCI_RangeAndBearingSensor::TReadings& tPackets = m_pcRABS->GetReadings();
    for(size_t i = 0; i < tPackets.size(); ++i) {
        UInt8 id;
        float prevDist = 100000.0f;
        CByteArray ba = tPackets[i].Data;
        ba >> id;
        SMessage* msg = NULL;
        auto search = m_sMessageTable.find(id);
        if(search != m_sMessageTable.end()) {
            prevDist = search->second->dist;
            msg = search->second;
        } else {
            msg = new SMessage();
            m_sMessageTable[id] = msg;
        }
        msg->id = id;
        ba >> msg->state;
        ba >> msg->fHop;
        ba >> msg->nHop;
        msg->dist  = (float)tPackets[i].Range;
        msg->angle = (float)tPackets[i].HorizontalBearing.GetValue();
        msg->prevDist = prevDist;
        msg->die = DIE_DOWN_RESET;
    }
    
    //reduce die counter and delete message if needs
    for(auto it = m_sMessageTable.begin(); it != m_sMessageTable.end(); ) {
        if(it->second->die > 0){
            //std::cout << GetId() << " reduce die counter:" << it->second->die << std::endl;
            it->second->die--;
            ++it;
        } else {
            //std::cout << GetId() << " reduce die counter - DEL:" << it->second->die << std::endl;
            delete it->second;
            it = m_sMessageTable.erase(it);
        }
    }
}

void CFootBotForaging::sendMessages() {
    CByteArray ba;
    if(m_sStateData.State == SStateData::STATE_WALKER) {
        ba << (UInt8)m_sStateData.id;
        ba << (UInt8)m_sStateData.State;
        ba << (int)MAX_HOP_CNT;        //MAX_HOP_CNT
        ba << (int)MAX_HOP_CNT;        //MAX_HOP_CNT
    } else {
        determineHopcount();
        ba << (UInt8)m_sStateData.id;
        ba << (UInt8)m_sStateData.State;
        ba << (int)m_sStateData.myFoodHop;
        ba << (int)m_sStateData.myNestHop;
    }
    m_pcRABA->SetData(ba);
}

size_t CFootBotForaging::determineHopcount() {
    size_t lowestNestHopFound = MAX_HOP_CNT;
    size_t lowestFoodHopFound = MAX_HOP_CNT;
    
    for( const auto& n : m_sMessageTable ) {
        if(n.second->nHop <= lowestNestHopFound)
            lowestNestHopFound = n.second->nHop;
        if(n.second->fHop <= lowestFoodHopFound)
            lowestFoodHopFound = n.second->fHop;
    }
    
    if(lowestNestHopFound < MAX_HOP_CNT)
        lowestNestHopFound += 1;
    if(lowestFoodHopFound < MAX_HOP_CNT)
        lowestFoodHopFound += 1;
    m_sStateData.myNestHop = lowestNestHopFound;
    m_sStateData.myFoodHop = lowestFoodHopFound;

#ifdef USE_FOOD_NEST_BEACON
    if(m_sStateData.State == SStateData::STATE_FOOD) 
        lowestFoodHopFound = 0;
    else if(m_sStateData.State == SStateData::STATE_NEST) 
        lowestNestHopFound = 0;
#else
    const CCI_PositioningSensor::SReading& sReading = m_pcPosSensor->GetReading();
    if((CVector2(sReading.Position[0], sReading.Position[1]) - m_cFoodPos).SquareLength() 
                            < m_sForagingParams.MaxDistanceDetectFoodNest) {
        lowestFoodHopFound = 0;
    } else if((CVector2(sReading.Position[0], sReading.Position[1]) - m_cNestPos).SquareLength() 
                            < m_sForagingParams.MaxDistanceDetectFoodNest)  {
        lowestNestHopFound = 0;
    }
#endif
    m_sStateData.myNestHop = lowestNestHopFound;
    m_sStateData.myFoodHop = lowestFoodHopFound;

    return 1;
}

bool CFootBotForaging::isItBeacon(size_t id) {
    // size_t least = 0;
    // size_t max = 15;
    // size_t nHop = m_sMessageTable[id].nHop;
    // size_t fHop = m_sMessageTable[id].fHop;
    // if((nHop > least && nHop < max) || (fHop > least && fHop < max))
        // return true;
    // else:
        // return false;
    return m_sMessageTable[id]->state == SStateData::STATE_BEACON 
           or m_sMessageTable[id]->state == SStateData::STATE_FOOD 
           or m_sMessageTable[id]->state == SStateData::STATE_NEST;
}

size_t CFootBotForaging::howManyBeacons() {
    size_t beacons = 0;
    size_t walkers = 0;
    size_t nOutRangeBeacons = 0;
    size_t maxDist = m_sForagingParams.MaxBeaconDistance;
    float  fMinDist = 1000.0f, fMaxDist = -1000.0f;
    for( const auto& n : m_sMessageTable ) {
        if(isItBeacon(n.first)) {
            //imposing restriction, beacons have to be within a certain distance, not just range of messaging
            if(n.second->dist < maxDist) {
                beacons += 1;
                fMinDist = std::min(n.second->dist, fMinDist);
                fMaxDist = std::max(n.second->dist, fMaxDist);
            } else {
                nOutRangeBeacons += 1;
            }
        } else {
            walkers++;
        }
    }
    //if(m_sStateData.State == SStateData::STATE_BEACON and m_sStateData.beaconPersistenceCnt == 0)
    //    std::cout << GetId() << ": iB=" << beacons << ", oB=" << nOutRangeBeacons 
    //              << ", m=" << fMinDist << ", M=" << fMaxDist << std::endl;
    m_sStateData.nWalkers = walkers;
    return beacons;
}

bool CFootBotForaging::shouldIExplore() {
    return (m_sStateData.nBeacons == 0);
}

bool CFootBotForaging::isObstacle() {
    bool bCollision;
    CVector2 cDiffusion = DiffusionVector(bCollision);
    return bCollision;
}

bool CFootBotForaging::acquireTarget(size_t type) {
    size_t tId = UNDEFINED_TARGET;
    size_t tempHop = MAX_HOP_CNT;
    size_t tempDist = 200;
    if(type == SStateData::GOAL_FOOD) {
        for( const auto& n : m_sMessageTable ) {
            if((n.second->die < DIE_DOWN_RESET / 2) or (not (isItBeacon(n.first) || n.second->state == SStateData::STATE_FOOD)))
                continue;
            if(n.second->fHop < tempHop) {
                tempHop = n.second->fHop;
                tId = n.first;
                tempDist = n.second->dist;
            } else if(n.second->fHop != MAX_HOP_CNT and n.second->fHop == tempHop and n.second->dist < tempDist) {
                tempHop = n.second->fHop;
                tId = n.first;
                tempDist = n.second->dist;
            }
        }
    } else if (type == SStateData::GOAL_NEST) {
        for( const auto& n : m_sMessageTable ) {
            if((n.second->die < DIE_DOWN_RESET / 2) or (not (isItBeacon(n.first) || n.second->state == SStateData::STATE_NEST)))
                continue;
            if(n.second->nHop < tempHop) {
                tempHop = n.second->nHop;
                tId = n.first;
                tempDist = n.second->dist;
            } else if(n.second->nHop != MAX_HOP_CNT and n.second->nHop == tempHop and n.second->dist < tempDist) {
                tempHop = n.second->nHop;
                tId = n.first;
                tempDist = n.second->dist;
            }
        }
    }
    m_sStateData.target_id = tId;
    if(tId != UNDEFINED_TARGET) 
        return true;
    else
        return false;
}

void CFootBotForaging::pursue() {
    auto search = m_sMessageTable.find(m_sStateData.target_id);
    //if target in sight of robot       self.target_id in self.messageTable
    if (search != m_sMessageTable.end()) {
        if(m_sStateData.pursueStepCounter > 0) {
            m_sStateData.pursueStepCounter -= 1;
            m_sStateData.currentpursueAction = SStateData::MOVE_FORWARD;
        } else {
            m_sStateData.pursueStepCounter = PURSUE_RESET;
            //if distance to target is increasing
            if(search->second->prevDist <= search->second->dist) {
                m_sStateData.currentpursueAction = SStateData::MOVE_TURN;
            } else {
                m_sStateData.currentpursueAction = SStateData::MOVE_FORWARD;
            }
        }
    } else {
        m_sStateData.Action = SStateData::ACTION_EXPLORE;
        m_sStateData.currentpursueAction = SStateData::MOVE_FORWARD;
    }
    m_sStateData.Movement = (SStateData::EMovement)m_sStateData.currentpursueAction;
}

/* 
 * 20200215 160900 ok but no_food_return not stable with Hoff
 *
void CFootBotForaging::runMotors() {
    if(m_sStateData.Movement == SStateData::MOVE_STATIONARY) {
        m_pcWheels->SetLinearVelocity(0.0f, 0.0f);
    } else {
        // Calculate diffusion vector and collision avoidance accordingly
        // Get the diffusion vector to perform obstacle avoidance
        bool bCollision;
        CVector2 cDiffusion = DiffusionVector(bCollision);
        auto search = m_sMessageTable.find(m_sStateData.target_id);
        if (search == m_sMessageTable.end() or bCollision) {    // or m_sStateData.Movement == SStateData::MOVE_TURN
            // Use the diffusion vector only //
            SetWheelSpeedsFromVector(m_sWheelTurningParams.MaxSpeed * cDiffusion);
        //} else if (m_sStateData.Movement == SStateData::MOVE_TURN) {
        //    SetWheelSpeedsFromVector(m_sWheelTurningParams.MaxSpeed * 0.80f * cDiffusion +
        //        m_sWheelTurningParams.MaxSpeed * 0.20f * CVector2(1.0f, CRadians::PI_OVER_FOUR));
        } else {
            if(m_sStateData.Action == SStateData::ACTION_EXPLORE) {
                SetWheelSpeedsFromVector(m_sWheelTurningParams.MaxSpeed * cDiffusion);
            } else {
#ifdef USE_FOOD_NEST_BEACON
                SetWheelSpeedsFromVector(m_sWheelTurningParams.MaxSpeed * 0.75f * cDiffusion +
                    m_sWheelTurningParams.MaxSpeed * 0.25f * CVector2(1.0f, search->second->angle));
#else
                SetWheelSpeedsFromVector(m_sWheelTurningParams.MaxSpeed * 0.80f * cDiffusion +
                    m_sWheelTurningParams.MaxSpeed * 0.20f * CVector2(1.0f, search->second->angle));
#endif
            }
        }
    }
}*/

/* 
 * 20200215 160900 ok but no_food_return not stable with Hoff
 */
void CFootBotForaging::runMotors() {
    if(m_sStateData.Movement == SStateData::MOVE_STATIONARY) {
        m_pcWheels->SetLinearVelocity(0.0f, 0.0f);
    } else {
        // Calculate diffusion vector and collision avoidance accordingly
        // Get the diffusion vector to perform obstacle avoidance
        bool bCollision;
        CVector2 cDiffusion = DiffusionVector(bCollision);
        CRadians baseTurnAngle = CRadians::PI_OVER_THREE;     //CRadians::PI_OVER_FOUR    PI_OVER_THREE     PI_OVER_TWO
        auto search = m_sMessageTable.find(m_sStateData.target_id);
        if(m_sStateData.Movement == SStateData::MOVE_FORWARD) {
            if (search == m_sMessageTable.end()) {
                //EXPLORE with no target - make turn of 45
                SetWheelSpeedsFromVector(m_sWheelTurningParams.MaxSpeed * cDiffusion);
            } else { 
                //EXPLORE with target
#ifdef USE_FOOD_NEST_BEACON
                SetWheelSpeedsFromVector(m_sWheelTurningParams.MaxSpeed * 0.75f * cDiffusion +
                    m_sWheelTurningParams.MaxSpeed * 0.25f * CVector2(1.0f, search->second->angle));
#else
                cDiffusion.Rotate(m_sForagingParams.TurnToTargetRate*CRadians(search->second->angle));
                SetWheelSpeedsFromVector(m_sWheelTurningParams.MaxSpeed * cDiffusion);
#endif
            }
        } else { 
            if (search == m_sMessageTable.end() /*or bCollision or m_sStateData.Movement == SStateData::MOVE_TURN*/) {
                //TURN without a target, make turn of 45
                Real prob = m_pcRNG->Uniform(CRange<Real>(0.0f, 1.0f));
                if(prob < 0.5f)
                    cDiffusion.Rotate(-baseTurnAngle);
                else
                    cDiffusion.Rotate(baseTurnAngle);
                SetWheelSpeedsFromVector(m_sWheelTurningParams.MaxSpeed * cDiffusion);
            } else {
                //TURN with target
#ifdef USE_FOOD_NEST_BEACON
                SetWheelSpeedsFromVector(m_sWheelTurningParams.MaxSpeed * 0.75f * cDiffusion +
                    m_sWheelTurningParams.MaxSpeed * 0.25f * CVector2(1.0f, search->second->angle));
#else
                cDiffusion.Rotate(m_sForagingParams.TurnToTargetRate*CRadians(search->second->angle));
                SetWheelSpeedsFromVector(m_sWheelTurningParams.MaxSpeed * cDiffusion);
#endif
            }
        }
    }
}

/****************************************/
/****************************************/
CVector2 CFootBotForaging::CalculateVectorToLight() {
    /* Get readings from light sensor */
    const CCI_FootBotLightSensor::TReadings& tLightReads = m_pcLight->GetReadings();
    /* Sum them together */
    CVector2 cAccumulator;
    for(size_t i = 0; i < tLightReads.size(); ++i) {
        cAccumulator += CVector2(tLightReads[i].Value, tLightReads[i].Angle);
    }
    /* If the light was perceived, return the vector */
    if(cAccumulator.Length() > 0.0f) {
        return CVector2(1.0f, cAccumulator.Angle());
    }
    /* Otherwise, return zero */
    else {
        return CVector2();
    }
}

/****************************************/
/****************************************/
CVector2 CFootBotForaging::DiffusionVector(bool& b_collision) {
    /* Get readings from proximity sensor */
    const CCI_FootBotProximitySensor::TReadings& tProxReads = m_pcProximity->GetReadings();
    /* Sum them together */
    CVector2 cDiffusionVector;
    for(size_t i = 0; i < tProxReads.size(); ++i) {
        cDiffusionVector += CVector2(tProxReads[i].Value, tProxReads[i].Angle);
    }
    /* If the angle of the vector is small enough and the closest obstacle
      is far enough, ignore the vector and go straight, otherwise return
      it */
    if(m_sDiffusionParams.GoStraightAngleRange.WithinMinBoundIncludedMaxBoundIncluded(cDiffusionVector.Angle()) &&
      cDiffusionVector.Length() < m_sDiffusionParams.Delta ) {
        b_collision = false;
        return CVector2::X;
    }
    else {
        b_collision = true;
        cDiffusionVector.Normalize();
        return -cDiffusionVector;
    }
}

/****************************************/
/****************************************/
void CFootBotForaging::SetWheelSpeedsFromVector(const CVector2& c_heading) {
    /* Get the heading angle */
    CRadians cHeadingAngle = c_heading.Angle().SignedNormalize();
    /* Get the length of the heading vector */
    Real fHeadingLength = c_heading.Length();
    /* Clamp the speed so that it's not greater than MaxSpeed */
    Real fBaseAngularWheelSpeed = Min<Real>(fHeadingLength, m_sWheelTurningParams.MaxSpeed);
    /* State transition logic */
    if(m_sWheelTurningParams.TurningMechanism == SWheelTurningParams::HARD_TURN) {
        if(Abs(cHeadingAngle) <= m_sWheelTurningParams.SoftTurnOnAngleThreshold) {
            m_sWheelTurningParams.TurningMechanism = SWheelTurningParams::SOFT_TURN;
        }
    }
    if(m_sWheelTurningParams.TurningMechanism == SWheelTurningParams::SOFT_TURN) {
        if(Abs(cHeadingAngle) > m_sWheelTurningParams.HardTurnOnAngleThreshold) {
            m_sWheelTurningParams.TurningMechanism = SWheelTurningParams::HARD_TURN;
        }
        else if(Abs(cHeadingAngle) <= m_sWheelTurningParams.NoTurnAngleThreshold) {
            m_sWheelTurningParams.TurningMechanism = SWheelTurningParams::NO_TURN;
        }
    }
    if(m_sWheelTurningParams.TurningMechanism == SWheelTurningParams::NO_TURN) {
        if(Abs(cHeadingAngle) > m_sWheelTurningParams.HardTurnOnAngleThreshold) {
            m_sWheelTurningParams.TurningMechanism = SWheelTurningParams::HARD_TURN;
        }
        else if(Abs(cHeadingAngle) > m_sWheelTurningParams.NoTurnAngleThreshold) {
            m_sWheelTurningParams.TurningMechanism = SWheelTurningParams::SOFT_TURN;
        }
    }
    /* Wheel speeds based on current turning state */
    Real fSpeed1, fSpeed2;
    switch(m_sWheelTurningParams.TurningMechanism) {
        case SWheelTurningParams::NO_TURN: {
            /* Just go straight */
            fSpeed1 = fBaseAngularWheelSpeed;
            fSpeed2 = fBaseAngularWheelSpeed;
            break;
        }
        case SWheelTurningParams::SOFT_TURN: {
            /* Both wheels go straight, but one is faster than the other */
            Real fSpeedFactor = (m_sWheelTurningParams.HardTurnOnAngleThreshold - 
                Abs(cHeadingAngle)) / m_sWheelTurningParams.HardTurnOnAngleThreshold;
            fSpeed1 = fBaseAngularWheelSpeed - fBaseAngularWheelSpeed * (1.0 - fSpeedFactor);
            fSpeed2 = fBaseAngularWheelSpeed + fBaseAngularWheelSpeed * (1.0 - fSpeedFactor);
            break;
        }
        case SWheelTurningParams::HARD_TURN: {
            /* Opposite wheel speeds */
            fSpeed1 = -m_sWheelTurningParams.MaxSpeed;
            fSpeed2 =  m_sWheelTurningParams.MaxSpeed;
            break;
        }
    }
    /* Apply the calculated speeds to the appropriate wheels */
    Real fLeftWheelSpeed, fRightWheelSpeed;
    if(cHeadingAngle > CRadians::ZERO) {
        /* Turn Left */
        fLeftWheelSpeed  = fSpeed1;
        fRightWheelSpeed = fSpeed2;
    }
    else {
        /* Turn Right */
        fLeftWheelSpeed  = fSpeed2;
        fRightWheelSpeed = fSpeed1;
    }
    /* Finally, set the wheel speeds 
    //if(GetId() == "fb0") {
    //    std::cout << GetId() << " new speed(" << fLeftWheelSpeed << "," << fRightWheelSpeed << ")" << std::endl;
    //}*/
    m_pcWheels->SetLinearVelocity(fLeftWheelSpeed, fRightWheelSpeed);
}

/****************************************/
/****************************************/
void CFootBotForaging::turnOnLights() {
    if(m_sStateData.State == SStateData::STATE_BEACON) {
        m_pcLEDs->SetAllColors(CColor::RED);
    } else if(m_sStateData.State == SStateData::STATE_WALKER) {
        if (m_sStateData.HasFood && m_sStateData.Goal == SStateData::GOAL_NEST) {
            m_pcLEDs->SetAllColors(CColor::GREEN);
        } else {
            m_pcLEDs->SetAllColors(CColor::BLUE);
        }
    } else if(m_sStateData.State == SStateData::STATE_FOOD) {
        m_pcLEDs->SetAllColors(CColor::BLACK);
    } else {
        m_pcLEDs->SetAllColors(CColor::GRAY50);
    }
}

/****************************************/
/****************************************/
/*
 * This statement notifies ARGoS of the existence of the controller.
 * It binds the class passed as first argument to the string passed as
 * second argument.
 * The string is then usable in the XML configuration file to refer to
 * this controller.
 * When ARGoS reads that string in the XML file, it knows which controller
 * class to instantiate.
 * See also the XML configuration files for an example of how this is used.
 */
REGISTER_CONTROLLER(CFootBotForaging, "footbot_foraging_controller")
