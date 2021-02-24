/*
 * Controller : OpenCM9.04 with 485 EXP board
 * Dynamixel : MX-28/64/106(with Protocol 2.0), X-series(except XL-320)
 * Power source : 12V SMPS to 485 EXP board(or 24V for Dynamixel Pro series)
 * 
 * Dynamixels are connected to Dynamixel BUS on 485 EXP board
 * http://emanual.robotis.com/docs/en/parts/controller/opencm485exp/#layout
*/

#include <DynamixelSDK.h>

// Control table address could be differ by Dynamixel Series
#define ADDRESS_TORQUE_ENABLE           24
#define ADDRESS_LED                     25
#define ADDRESS_GOAL_POSITION           30
#define ADDRESS_PRESENT_POSITION        36

#define ADDR_AX_TORQUE_ENABLE           24                 // Control table address is different in Dynamixel model
#define ADDR_AX_GOAL_POSITION           30
#define ADDR_AX_PRESENT_POSITION        36

// Data Byte Length
#define LENGTH_LED                      1
#define LENGTH_GOAL_POSITION            2
#define LENGTH_PRESENT_POSITION         2

// Protocol version
#define PROTOCOL_VERSION                1.0                 // See which protocol version is used in the Dynamixel

// Default setting mx
#define DXL1_ID                         3                   
#define DXL2_ID                         4                   
#define DXL3_ID                         5                   
#define DXL4_ID                         6                   
#define DXL5_ID                         7                   
#define DXL6_ID                         9                   
#define DXL7_ID                         10                   
#define DXL8_ID                         11
#define DXL9_ID                         12                   
#define DXL10_ID                         13                  
                  
#define BAUDRATE                        1000000
#define DEVICENAME                      "1"                 

// Default setting ax
#define DXL_ID14                          14
#define DXL_ID15                         15
#define DXL_ID16                          16
#define DXL_ID17                          17
#define DXL_ID18                          18
#define DXL_ID19                          19
#define DXL_ID20                          20
#define DXL_ID21                          21
#define DXL_ID22                          22
#define BAUDRATEax                        1000000
#define DEVICENAMEax                      "1"                 //DEVICENAME "1" -> Serial1(OpenCM9.04 DXL TTL Ports)
                                                            

#define TORQUE_ENABLE                   1                   // Value for enabling the torque
#define TORQUE_DISABLE                  0                   // Value for disabling the torque
#define DXL_MINIMUM_POSITION_VALUE      100                 // Dynamixel will rotate between this value
#define DXL_MAXIMUM_POSITION_VALUE      4000                // and this value (note that the Dynamixel would not move when the position value is out of movable range. Check e-manual about the range of the Dynamixel you use.)
#define DXL_MINIMUM_POSITION_VALUEax      100                 // Dynamixel will rotate between this value
#define DXL_MAXIMUM_POSITION_VALUEax      1000
#define DXL_MOVING_STATUS_THRESHOLD     20                  // Dynamixel moving status threshold

#define ESC_ASCII_VALUE                 0x1b



void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  while(!Serial);


  Serial.println("Start..");

   // Initialize PortHandler instance
  // Set the port path
  // Get methods and members of PortHandlerLinux or PortHandlerWindows
  dynamixel::PortHandler *portHandler = dynamixel::PortHandler::getPortHandler(DEVICENAME);

  // Initialize PacketHandler instance
  // Set the protocol version
  // Get methods and members of Protocol1PacketHandler or Protocol2PacketHandler
  dynamixel::PacketHandler *packetHandler = dynamixel::PacketHandler::getPacketHandler(PROTOCOL_VERSION);

  dynamixel::PortHandler *portHandlerax = dynamixel::PortHandler::getPortHandler(DEVICENAMEax);
  // Initialize GroupBulkWrite instance
  dynamixel::GroupBulkWrite groupBulkWrite(portHandler, packetHandler);

  dynamixel::PacketHandler *packetHandlerax = dynamixel::PacketHandler::getPacketHandler(PROTOCOL_VERSION);
  // Initialize GroupBulkRead instance
  dynamixel::GroupBulkRead groupBulkRead(portHandler, packetHandler);

  int index = 0;
  int index14 = 0;
  int index15 = 0;
  int index16 = 0;
  int index17 = 0;
  int index18 = 0;
  int index19 = 0;
  int index20 = 0;
  int index21 = 0;
  int index22 = 0;
  int dxl_comm_result = COMM_TX_FAIL;             // Communication result
  int dxl_comm_result14 = COMM_TX_FAIL;             
  int dxl_comm_result15 = COMM_TX_FAIL;             
  int dxl_comm_result16 = COMM_TX_FAIL;             
  int dxl_comm_result17 = COMM_TX_FAIL;             
  int dxl_comm_result18 = COMM_TX_FAIL;             
  int dxl_comm_result19 = COMM_TX_FAIL;             
  int dxl_comm_result20 = COMM_TX_FAIL;             
  int dxl_comm_result21 = COMM_TX_FAIL;             
  int dxl_comm_result22 = COMM_TX_FAIL;
  bool dxl_addparam_result = false;                // addParam result
  bool dxl_getdata_result = false;                 // GetParam result
  int dxl_goal_position[2] = {DXL_MINIMUM_POSITION_VALUE, DXL_MAXIMUM_POSITION_VALUE};         // Goal position
  int dxl_goal_position14[2] = {DXL_MINIMUM_POSITION_VALUEax, DXL_MAXIMUM_POSITION_VALUEax};         // Goal position
  int dxl_goal_position15[2] = {DXL_MINIMUM_POSITION_VALUEax, DXL_MAXIMUM_POSITION_VALUEax};         // Goal position
  int dxl_goal_position16[2] = {DXL_MINIMUM_POSITION_VALUEax, DXL_MAXIMUM_POSITION_VALUEax};         // Goal position
  int dxl_goal_position17[2] = {DXL_MINIMUM_POSITION_VALUEax, DXL_MAXIMUM_POSITION_VALUEax};         // Goal position
  int dxl_goal_position18[2] = {DXL_MINIMUM_POSITION_VALUEax, DXL_MAXIMUM_POSITION_VALUEax};         // Goal position
  int dxl_goal_position19[2] = {DXL_MINIMUM_POSITION_VALUEax, DXL_MAXIMUM_POSITION_VALUEax};         // Goal position
  int dxl_goal_position20[2] = {DXL_MINIMUM_POSITION_VALUEax, DXL_MAXIMUM_POSITION_VALUEax};         // Goal position
  int dxl_goal_position21[2] = {DXL_MINIMUM_POSITION_VALUEax, DXL_MAXIMUM_POSITION_VALUEax};         // Goal position
  int dxl_goal_position22[2] = {DXL_MINIMUM_POSITION_VALUEax, DXL_MAXIMUM_POSITION_VALUEax};         // Goal position


  uint8_t dxl_error = 0;                          // Dynamixel error
  uint8_t dxl_led_value[2] = {0x00, 0x01};        // Dynamixel LED value for write
  uint8_t param_goal_position[4];
  int32_t dxl1_present_position = 0;              // Present position
  int32_t dxl2_present_position = 0;              // Present position
  int32_t dxl3_present_position = 0;              // Present position
  int32_t dxl4_present_position = 0;              // Present position
  int32_t dxl5_present_position = 0;              // Present position
  int32_t dxl6_present_position = 0;              // Present position
  int32_t dxl7_present_position = 0;              // Present position
  int32_t dxl8_present_position = 0;              // Present position
  int32_t dxl9_present_position = 0;              // Present position
  int32_t dxl10_present_position = 0;              // Present position
  int16_t dxl_present_position14 = 0;               // Present position
  int16_t dxl_present_position15 = 0;               // Present position
  int16_t dxl_present_position16 = 0;               // Present position
  int16_t dxl_present_position17 = 0;               // Present position
  int16_t dxl_present_position18 = 0;               // Present position
  int16_t dxl_present_position19 = 0;               // Present position
  int16_t dxl_present_position20 = 0;               // Present position
  int16_t dxl_present_position21 = 0;               // Present position
  int16_t dxl_present_position22 = 0;               // Present position
  uint8_t dxl2_led_value_read;                    // Dynamixel LED value for read

  // Open port
  if (portHandler->openPort())
  {
    Serial.print("Succeeded to open the port!\n");
  }
  else
  {
    Serial.print("Failed to open the port!\n");
    Serial.print("Press any key to terminate...\n");
    return;
  }

  // Set port baudrate
  if (portHandler->setBaudRate(BAUDRATE))
  {
    Serial.print("Succeeded to change the baudrate!\n");
  }
  else
  {
    Serial.print("Failed to change the baudrate!\n");
    Serial.print("Press any key to terminate...\n");
    return;
  }
  if (portHandlerax->openPort())
  {
    Serial.print("Succeeded to open the port!\n");
  }
  else
  {
    Serial.print("Failed to open the port!\n");
    Serial.print("Press any key to terminate...\n");
    return;
  }

  // Set port baudrate
  if (portHandlerax->setBaudRate(BAUDRATEax))
  {
    Serial.print("Succeeded to change the baudrate!\n");
  }
  else
  {
    Serial.print("Failed to change the baudrate!\n");
    Serial.print("Press any key to terminate...\n");
    return;
  }

  // Enable Dynamixel#1 Torque
  dxl_comm_result = packetHandler->write1ByteTxRx(portHandler, DXL1_ID, ADDRESS_TORQUE_ENABLE, TORQUE_ENABLE, &dxl_error);
  dxl_comm_result14 = packetHandlerax->write1ByteTxRx(portHandlerax, DXL_ID14, ADDR_AX_TORQUE_ENABLE, TORQUE_ENABLE, &dxl_error);
  dxl_comm_result15 = packetHandlerax->write1ByteTxRx(portHandlerax, DXL_ID15, ADDR_AX_TORQUE_ENABLE, TORQUE_ENABLE, &dxl_error);
  dxl_comm_result16 = packetHandlerax->write1ByteTxRx(portHandlerax, DXL_ID16, ADDR_AX_TORQUE_ENABLE, TORQUE_ENABLE, &dxl_error);
  dxl_comm_result17 = packetHandlerax->write1ByteTxRx(portHandlerax, DXL_ID17, ADDR_AX_TORQUE_ENABLE, TORQUE_ENABLE, &dxl_error);
  dxl_comm_result18 = packetHandlerax->write1ByteTxRx(portHandlerax, DXL_ID18, ADDR_AX_TORQUE_ENABLE, TORQUE_ENABLE, &dxl_error);
  dxl_comm_result19 = packetHandlerax->write1ByteTxRx(portHandlerax, DXL_ID19, ADDR_AX_TORQUE_ENABLE, TORQUE_ENABLE, &dxl_error);
  dxl_comm_result20 = packetHandlerax->write1ByteTxRx(portHandlerax, DXL_ID20, ADDR_AX_TORQUE_ENABLE, TORQUE_ENABLE, &dxl_error);
  dxl_comm_result21 = packetHandlerax->write1ByteTxRx(portHandlerax, DXL_ID21, ADDR_AX_TORQUE_ENABLE, TORQUE_ENABLE, &dxl_error);
  dxl_comm_result22 = packetHandlerax->write1ByteTxRx(portHandlerax, DXL_ID22, ADDR_AX_TORQUE_ENABLE, TORQUE_ENABLE, &dxl_error);

   if (dxl_comm_result14 != COMM_SUCCESS)
  {
    packetHandlerax->getTxRxResult(dxl_comm_result14);
  }
  else if (dxl_error != 0)
  {
    packetHandlerax->getRxPacketError(dxl_error);
  }
  else if (dxl_comm_result15 != COMM_SUCCESS)
  {
    packetHandlerax->getTxRxResult(dxl_comm_result15);
  }
  else if (dxl_error != 0)
  {
    packetHandlerax->getRxPacketError(dxl_error);
  }
  else if (dxl_comm_result16 != COMM_SUCCESS)
  {
    packetHandlerax->getTxRxResult(dxl_comm_result16);
  }
  else if (dxl_error != 0)
  {
    packetHandlerax->getRxPacketError(dxl_error);
  }
  else if (dxl_comm_result17 != COMM_SUCCESS)
  {
    packetHandlerax->getTxRxResult(dxl_comm_result17);
  }
  else if (dxl_error != 0)
  {
    packetHandlerax->getRxPacketError(dxl_error);
  }
  else if (dxl_comm_result18 != COMM_SUCCESS)
  {
    packetHandlerax->getTxRxResult(dxl_comm_result18);
  }
  else if (dxl_error != 0)
  {
    packetHandlerax->getRxPacketError(dxl_error);
  }
  else if (dxl_comm_result19 != COMM_SUCCESS)
  {
    packetHandlerax->getTxRxResult(dxl_comm_result19);
  }
  else if (dxl_error != 0)
  {
    packetHandlerax->getRxPacketError(dxl_error);
  }
  else if (dxl_comm_result20 != COMM_SUCCESS)
  {
    packetHandlerax->getTxRxResult(dxl_comm_result20);
  }
  else if (dxl_error != 0)
  {
    packetHandlerax->getRxPacketError(dxl_error);
  }
  else if (dxl_comm_result21 != COMM_SUCCESS)
  {
    packetHandlerax->getTxRxResult(dxl_comm_result21);
  }
  else if (dxl_error != 0)
  {
    packetHandlerax->getRxPacketError(dxl_error);
  }
  else if (dxl_comm_result22 != COMM_SUCCESS)
  {
    packetHandlerax->getTxRxResult(dxl_comm_result22);
  }
  else if (dxl_error != 0)
  {
    packetHandlerax->getRxPacketError(dxl_error);
  }
  
  if (dxl_comm_result != COMM_SUCCESS)
  {
    packetHandler->getTxRxResult(dxl_comm_result);
  }
  else if (dxl_error != 0)
  {
    packetHandler->getRxPacketError(dxl_error);
  }

  // Enable Dynamixel#2 Torque
  dxl_comm_result = packetHandler->write1ByteTxRx(portHandler, DXL2_ID, ADDRESS_TORQUE_ENABLE, TORQUE_ENABLE, &dxl_error);
  if (dxl_comm_result != COMM_SUCCESS)
  {
    packetHandler->getTxRxResult(dxl_comm_result);
  }
  else if (dxl_error != 0)
  {
    packetHandler->getRxPacketError(dxl_error);
  }

  // Enable Dynamixel#3 Torque
  dxl_comm_result = packetHandler->write1ByteTxRx(portHandler, DXL3_ID, ADDRESS_TORQUE_ENABLE, TORQUE_ENABLE, &dxl_error);
  if (dxl_comm_result != COMM_SUCCESS)
  {
    packetHandler->getTxRxResult(dxl_comm_result);
  }
  else if (dxl_error != 0)
  {
    packetHandler->getRxPacketError(dxl_error);
  }

  // Enable Dynamixel#4 Torque
  dxl_comm_result = packetHandler->write1ByteTxRx(portHandler, DXL4_ID, ADDRESS_TORQUE_ENABLE, TORQUE_ENABLE, &dxl_error);
  if (dxl_comm_result != COMM_SUCCESS)
  {
    packetHandler->getTxRxResult(dxl_comm_result);
  }
  else if (dxl_error != 0)
  {
    packetHandler->getRxPacketError(dxl_error);
  }

  // Enable Dynamixel#5 Torque
  dxl_comm_result = packetHandler->write1ByteTxRx(portHandler, DXL5_ID, ADDRESS_TORQUE_ENABLE, TORQUE_ENABLE, &dxl_error);
  if (dxl_comm_result != COMM_SUCCESS)
  {
    packetHandler->getTxRxResult(dxl_comm_result);
  }
  else if (dxl_error != 0)
  {
    packetHandler->getRxPacketError(dxl_error);
  }
// Enable Dynamixel#6 Torque
  dxl_comm_result = packetHandler->write1ByteTxRx(portHandler, DXL6_ID, ADDRESS_TORQUE_ENABLE, TORQUE_ENABLE, &dxl_error);
  if (dxl_comm_result != COMM_SUCCESS)
  {
    packetHandler->getTxRxResult(dxl_comm_result);
  }
  else if (dxl_error != 0)
  {
    packetHandler->getRxPacketError(dxl_error);
  }
// Enable Dynamixel#7 Torque
  dxl_comm_result = packetHandler->write1ByteTxRx(portHandler, DXL7_ID, ADDRESS_TORQUE_ENABLE, TORQUE_ENABLE, &dxl_error);
  if (dxl_comm_result != COMM_SUCCESS)
  {
    packetHandler->getTxRxResult(dxl_comm_result);
  }
  else if (dxl_error != 0)
  {
    packetHandler->getRxPacketError(dxl_error);
  }
// Enable Dynamixel#8 Torque
  dxl_comm_result = packetHandler->write1ByteTxRx(portHandler, DXL8_ID, ADDRESS_TORQUE_ENABLE, TORQUE_ENABLE, &dxl_error);
  if (dxl_comm_result != COMM_SUCCESS)
  {
    packetHandler->getTxRxResult(dxl_comm_result);
  }
  else if (dxl_error != 0)
  {
    packetHandler->getRxPacketError(dxl_error);
  }
// Enable Dynamixel#9 Torque
  dxl_comm_result = packetHandler->write1ByteTxRx(portHandler, DXL9_ID, ADDRESS_TORQUE_ENABLE, TORQUE_ENABLE, &dxl_error);
  if (dxl_comm_result != COMM_SUCCESS)
  {
    packetHandler->getTxRxResult(dxl_comm_result);
  }
  else if (dxl_error != 0)
  {
    packetHandler->getRxPacketError(dxl_error);
  }
// Enable Dynamixel#10 Torque
  dxl_comm_result = packetHandler->write1ByteTxRx(portHandler, DXL10_ID, ADDRESS_TORQUE_ENABLE, TORQUE_ENABLE, &dxl_error);
  if (dxl_comm_result != COMM_SUCCESS)
  {
    packetHandler->getTxRxResult(dxl_comm_result);
  }
  else if (dxl_error != 0)
  {
    packetHandler->getRxPacketError(dxl_error);
  }

  // Add parameter storage for Dynamixel#1 present position
  dxl_addparam_result = groupBulkRead.addParam(DXL1_ID, ADDRESS_PRESENT_POSITION, LENGTH_PRESENT_POSITION);

  // Add parameter storage for Dynamixel#2 present position
  dxl_addparam_result = groupBulkRead.addParam(DXL2_ID, ADDRESS_PRESENT_POSITION, LENGTH_PRESENT_POSITION);

  // Add parameter storage for Dynamixel#3 present position
  dxl_addparam_result = groupBulkRead.addParam(DXL3_ID, ADDRESS_PRESENT_POSITION, LENGTH_PRESENT_POSITION);

  // Add parameter storage for Dynamixel#4 present position
  dxl_addparam_result = groupBulkRead.addParam(DXL4_ID, ADDRESS_PRESENT_POSITION, LENGTH_PRESENT_POSITION);

  // Add parameter storage for Dynamixel#5 present position
  dxl_addparam_result = groupBulkRead.addParam(DXL5_ID, ADDRESS_PRESENT_POSITION, LENGTH_PRESENT_POSITION);

  // Add parameter storage for Dynamixel#6 present position
  dxl_addparam_result = groupBulkRead.addParam(DXL6_ID, ADDRESS_PRESENT_POSITION, LENGTH_PRESENT_POSITION);

  // Add parameter storage for Dynamixel#7 present position
  dxl_addparam_result = groupBulkRead.addParam(DXL7_ID, ADDRESS_PRESENT_POSITION, LENGTH_PRESENT_POSITION);

  // Add parameter storage for Dynamixel#8 present position
  dxl_addparam_result = groupBulkRead.addParam(DXL8_ID, ADDRESS_PRESENT_POSITION, LENGTH_PRESENT_POSITION);

  // Add parameter storage for Dynamixel#9 present position
  dxl_addparam_result = groupBulkRead.addParam(DXL9_ID, ADDRESS_PRESENT_POSITION, LENGTH_PRESENT_POSITION);
  
// Add parameter storage for Dynamixel#10 present position
  dxl_addparam_result = groupBulkRead.addParam(DXL10_ID, ADDRESS_PRESENT_POSITION, LENGTH_PRESENT_POSITION);
  
  // Add parameter storage for Dynamixel#2 LED value
  dxl_addparam_result = groupBulkRead.addParam(DXL2_ID, ADDRESS_LED, LENGTH_LED);
  if (dxl_error != 0)
  {
    packetHandlerax->getRxPacketError(dxl_error);
  }
  else
  {
    Serial.print("Dynamixel has been successfully connected \n");
  }
  while(1)
  {
    Serial.print("Press any key to continue! (or press q to quit!)\n");


    while(Serial.available()==0);

    int ch;

    ch = Serial.read();
    if( ch == 'q' )
      break;

    // Allocate goal position value into byte array
    param_goal_position[0] = DXL_LOBYTE(DXL_LOWORD(dxl_goal_position[index]));
    param_goal_position[1] = DXL_HIBYTE(DXL_LOWORD(dxl_goal_position[index]));
    param_goal_position[2] = DXL_LOBYTE(DXL_HIWORD(dxl_goal_position[index]));
    param_goal_position[3] = DXL_HIBYTE(DXL_HIWORD(dxl_goal_position[index]));

     dxl_comm_result14 = packetHandlerax->write2ByteTxRx(portHandlerax, DXL_ID14, ADDR_AX_GOAL_POSITION, dxl_goal_position14[index14], &dxl_error);
    dxl_comm_result15 = packetHandlerax->write2ByteTxRx(portHandlerax, DXL_ID15, ADDR_AX_GOAL_POSITION, dxl_goal_position15[index15], &dxl_error);
    dxl_comm_result16 = packetHandlerax->write2ByteTxRx(portHandlerax, DXL_ID16, ADDR_AX_GOAL_POSITION, dxl_goal_position16[index16], &dxl_error);
    dxl_comm_result17 = packetHandlerax->write2ByteTxRx(portHandlerax, DXL_ID17, ADDR_AX_GOAL_POSITION, dxl_goal_position17[index17], &dxl_error);
    dxl_comm_result18 = packetHandlerax->write2ByteTxRx(portHandlerax, DXL_ID18, ADDR_AX_GOAL_POSITION, dxl_goal_position18[index18], &dxl_error);
    dxl_comm_result19 = packetHandlerax->write2ByteTxRx(portHandlerax, DXL_ID19, ADDR_AX_GOAL_POSITION, dxl_goal_position19[index19], &dxl_error);
    dxl_comm_result20 = packetHandlerax->write2ByteTxRx(portHandlerax, DXL_ID20, ADDR_AX_GOAL_POSITION, dxl_goal_position20[index20], &dxl_error);
    dxl_comm_result21 = packetHandlerax->write2ByteTxRx(portHandlerax, DXL_ID21, ADDR_AX_GOAL_POSITION, dxl_goal_position21[index21], &dxl_error);
    dxl_comm_result22 = packetHandlerax->write2ByteTxRx(portHandlerax, DXL_ID22, ADDR_AX_GOAL_POSITION, dxl_goal_position22[index22], &dxl_error);
    
    
    if (dxl_comm_result14 != COMM_SUCCESS)
    {
      packetHandlerax->getTxRxResult(dxl_comm_result14);
    }
    else if (dxl_error != 0)
    {
      packetHandlerax->getRxPacketError(dxl_error);
    }
    if (dxl_comm_result15 != COMM_SUCCESS)
    {
      packetHandlerax->getTxRxResult(dxl_comm_result15);
    }
    else if (dxl_error != 0)
    {
      packetHandlerax->getRxPacketError(dxl_error);
    }
    if (dxl_comm_result16 != COMM_SUCCESS)
    {
      packetHandlerax->getTxRxResult(dxl_comm_result16);
    }
    else if (dxl_error != 0)
    {
      packetHandlerax->getRxPacketError(dxl_error);
    }
    if (dxl_comm_result17 != COMM_SUCCESS)
    {
      packetHandlerax->getTxRxResult(dxl_comm_result17);
    }
    else if (dxl_error != 0)
    {
      packetHandlerax->getRxPacketError(dxl_error);
    }
    if (dxl_comm_result18 != COMM_SUCCESS)
    {
      packetHandlerax->getTxRxResult(dxl_comm_result18);
    }
    else if (dxl_error != 0)
    {
      packetHandlerax->getRxPacketError(dxl_error);
    }
    if (dxl_comm_result19 != COMM_SUCCESS)
    {
      packetHandlerax->getTxRxResult(dxl_comm_result19);
    }
    else if (dxl_error != 0)
    {
      packetHandlerax->getRxPacketError(dxl_error);
    }
    if (dxl_comm_result20 != COMM_SUCCESS)
    {
      packetHandlerax->getTxRxResult(dxl_comm_result20);
    }
    else if (dxl_error != 0)
    {
      packetHandlerax->getRxPacketError(dxl_error);
    }
    if (dxl_comm_result21 != COMM_SUCCESS)
    {
      packetHandlerax->getTxRxResult(dxl_comm_result21);
    }
    else if (dxl_error != 0)
    {
      packetHandlerax->getRxPacketError(dxl_error);
    }
    if (dxl_comm_result22 != COMM_SUCCESS)
    {
      packetHandlerax->getTxRxResult(dxl_comm_result22);
    }
    else if (dxl_error != 0)
    {
      packetHandlerax->getRxPacketError(dxl_error);
    }
    
    // Add parameter storage for Dynamixel#1 goal position
    dxl_addparam_result = groupBulkWrite.addParam(DXL1_ID, ADDRESS_GOAL_POSITION, LENGTH_GOAL_POSITION, param_goal_position);

    // Add parameter storage for Dynamixel#2 goal position
    dxl_addparam_result = groupBulkWrite.addParam(DXL2_ID, ADDRESS_GOAL_POSITION, LENGTH_GOAL_POSITION, param_goal_position);

    // Add parameter storage for Dynamixel#3 goal position
    dxl_addparam_result = groupBulkWrite.addParam(DXL3_ID, ADDRESS_GOAL_POSITION, LENGTH_GOAL_POSITION, param_goal_position);

    // Add parameter storage for Dynamixel#4 goal position
    dxl_addparam_result = groupBulkWrite.addParam(DXL4_ID, ADDRESS_GOAL_POSITION, LENGTH_GOAL_POSITION, param_goal_position);

    // Add parameter storage for Dynamixel#5 goal position
    dxl_addparam_result = groupBulkWrite.addParam(DXL5_ID, ADDRESS_GOAL_POSITION, LENGTH_GOAL_POSITION, param_goal_position);

    // Add parameter storage for Dynamixel#6 goal position
    dxl_addparam_result = groupBulkWrite.addParam(DXL6_ID, ADDRESS_GOAL_POSITION, LENGTH_GOAL_POSITION, param_goal_position);

    // Add parameter storage for Dynamixel#7 goal position
    dxl_addparam_result = groupBulkWrite.addParam(DXL7_ID, ADDRESS_GOAL_POSITION, LENGTH_GOAL_POSITION, param_goal_position);

    // Add parameter storage for Dynamixel#8 goal position
    dxl_addparam_result = groupBulkWrite.addParam(DXL8_ID, ADDRESS_GOAL_POSITION, LENGTH_GOAL_POSITION, param_goal_position);
    
    // Add parameter storage for Dynamixel#9 goal position
    dxl_addparam_result = groupBulkWrite.addParam(DXL9_ID, ADDRESS_GOAL_POSITION, LENGTH_GOAL_POSITION, param_goal_position);
    
    // Add parameter storage for Dynamixel#10 goal position
    dxl_addparam_result = groupBulkWrite.addParam(DXL10_ID, ADDRESS_GOAL_POSITION, LENGTH_GOAL_POSITION, param_goal_position);

    // Add parameter storage for Dynamixel#2 LED value
    dxl_addparam_result = groupBulkWrite.addParam(DXL2_ID, ADDRESS_LED, LENGTH_LED, &dxl_led_value[index]);

    // Bulkwrite goal position and LED value
    dxl_comm_result = groupBulkWrite.txPacket();
    if (dxl_comm_result != COMM_SUCCESS) packetHandler->getTxRxResult(dxl_comm_result);

    // Clear bulkwrite parameter storage
    groupBulkWrite.clearParam();

    do
    {
      // Bulkread present position and LED status
      dxl_comm_result = groupBulkRead.txRxPacket();
      if (dxl_comm_result != COMM_SUCCESS) packetHandler->getTxRxResult(dxl_comm_result);

      // Check if groupbulkread data of Dynamixel#1 is available
      dxl_getdata_result = groupBulkRead.isAvailable(DXL1_ID, ADDRESS_PRESENT_POSITION, LENGTH_PRESENT_POSITION);

      // Check if groupbulkread data of Dynamixel#2 is available
      dxl_getdata_result = groupBulkRead.isAvailable(DXL2_ID, ADDRESS_PRESENT_POSITION, LENGTH_PRESENT_POSITION);

      // Check if groupbulkread data of Dynamixel#3 is available
      dxl_getdata_result = groupBulkRead.isAvailable(DXL3_ID, ADDRESS_PRESENT_POSITION, LENGTH_PRESENT_POSITION);

      // Check if groupbulkread data of Dynamixel#4 is available
      dxl_getdata_result = groupBulkRead.isAvailable(DXL4_ID, ADDRESS_PRESENT_POSITION, LENGTH_PRESENT_POSITION);

      // Check if groupbulkread data of Dynamixel#5 is available
      dxl_getdata_result = groupBulkRead.isAvailable(DXL5_ID, ADDRESS_PRESENT_POSITION, LENGTH_PRESENT_POSITION);

      // Check if groupbulkread data of Dynamixel#6 is available
      dxl_getdata_result = groupBulkRead.isAvailable(DXL6_ID, ADDRESS_PRESENT_POSITION, LENGTH_PRESENT_POSITION);

      // Check if groupbulkread data of Dynamixel#7 is available
      dxl_getdata_result = groupBulkRead.isAvailable(DXL7_ID, ADDRESS_PRESENT_POSITION, LENGTH_PRESENT_POSITION);

      // Check if groupbulkread data of Dynamixel#8 is available
      dxl_getdata_result = groupBulkRead.isAvailable(DXL8_ID, ADDRESS_PRESENT_POSITION, LENGTH_PRESENT_POSITION);
      
      // Check if groupbulkread data of Dynamixel#9 is available
      dxl_getdata_result = groupBulkRead.isAvailable(DXL9_ID, ADDRESS_PRESENT_POSITION, LENGTH_PRESENT_POSITION);
      
      // Check if groupbulkread data of Dynamixel#10 is available
      dxl_getdata_result = groupBulkRead.isAvailable(DXL10_ID, ADDRESS_PRESENT_POSITION, LENGTH_PRESENT_POSITION);

      // Check if groupbulkread data of Dynamixel#2 is available
      dxl_getdata_result = groupBulkRead.isAvailable(DXL2_ID, ADDRESS_LED, LENGTH_LED);

      // Get present position value
      dxl1_present_position = groupBulkRead.getData(DXL1_ID, ADDRESS_PRESENT_POSITION, LENGTH_PRESENT_POSITION);

      // Get present position value
      dxl2_present_position = groupBulkRead.getData(DXL2_ID, ADDRESS_PRESENT_POSITION, LENGTH_PRESENT_POSITION);

      // Get present position value
      dxl3_present_position = groupBulkRead.getData(DXL3_ID, ADDRESS_PRESENT_POSITION, LENGTH_PRESENT_POSITION);

      // Get present position value
      dxl4_present_position = groupBulkRead.getData(DXL4_ID, ADDRESS_PRESENT_POSITION, LENGTH_PRESENT_POSITION);

      // Get present position value
      dxl5_present_position = groupBulkRead.getData(DXL5_ID, ADDRESS_PRESENT_POSITION, LENGTH_PRESENT_POSITION);

      // Get present position value
      dxl6_present_position = groupBulkRead.getData(DXL6_ID, ADDRESS_PRESENT_POSITION, LENGTH_PRESENT_POSITION);

      // Get present position value
      dxl7_present_position = groupBulkRead.getData(DXL7_ID, ADDRESS_PRESENT_POSITION, LENGTH_PRESENT_POSITION);

      // Get present position value
      dxl8_present_position = groupBulkRead.getData(DXL8_ID, ADDRESS_PRESENT_POSITION, LENGTH_PRESENT_POSITION);

      // Get present position value
      dxl9_present_position = groupBulkRead.getData(DXL9_ID, ADDRESS_PRESENT_POSITION, LENGTH_PRESENT_POSITION);

      // Get present position value
      dxl10_present_position = groupBulkRead.getData(DXL10_ID, ADDRESS_PRESENT_POSITION, LENGTH_PRESENT_POSITION);

     dxl_comm_result14 = packetHandlerax->read2ByteTxRx(portHandlerax, DXL_ID14, ADDR_AX_PRESENT_POSITION, (uint16_t*)&dxl_present_position14, &dxl_error);
      dxl_comm_result15 = packetHandlerax->read2ByteTxRx(portHandlerax, DXL_ID15, ADDR_AX_PRESENT_POSITION, (uint16_t*)&dxl_present_position15, &dxl_error);
      dxl_comm_result16 = packetHandlerax->read2ByteTxRx(portHandlerax, DXL_ID16, ADDR_AX_PRESENT_POSITION, (uint16_t*)&dxl_present_position16, &dxl_error);
      dxl_comm_result17 = packetHandlerax->read2ByteTxRx(portHandlerax, DXL_ID17, ADDR_AX_PRESENT_POSITION, (uint16_t*)&dxl_present_position17, &dxl_error);
      dxl_comm_result18 = packetHandlerax->read2ByteTxRx(portHandlerax, DXL_ID18, ADDR_AX_PRESENT_POSITION, (uint16_t*)&dxl_present_position18, &dxl_error);
      dxl_comm_result19 = packetHandlerax->read2ByteTxRx(portHandlerax, DXL_ID19, ADDR_AX_PRESENT_POSITION, (uint16_t*)&dxl_present_position19, &dxl_error);
      dxl_comm_result20 = packetHandlerax->read2ByteTxRx(portHandlerax, DXL_ID20, ADDR_AX_PRESENT_POSITION, (uint16_t*)&dxl_present_position20, &dxl_error);
      dxl_comm_result21 = packetHandlerax->read2ByteTxRx(portHandlerax, DXL_ID21, ADDR_AX_PRESENT_POSITION, (uint16_t*)&dxl_present_position21, &dxl_error);
      dxl_comm_result22 = packetHandlerax->read2ByteTxRx(portHandlerax, DXL_ID22, ADDR_AX_PRESENT_POSITION, (uint16_t*)&dxl_present_position22, &dxl_error);
      
      
      if (dxl_comm_result14 != COMM_SUCCESS)
      {
        packetHandlerax->getTxRxResult(dxl_comm_result14);
      }
      else if (dxl_error != 0)
      {
        packetHandlerax->getRxPacketError(dxl_error);
      }
      if (dxl_comm_result15 != COMM_SUCCESS)
      {
        packetHandlerax->getTxRxResult(dxl_comm_result15);
      }
      else if (dxl_error != 0)
      {
        packetHandlerax->getRxPacketError(dxl_error);
      }
      if (dxl_comm_result16 != COMM_SUCCESS)
      {
        packetHandlerax->getTxRxResult(dxl_comm_result16);
      }
      else if (dxl_error != 0)
      {
        packetHandlerax->getRxPacketError(dxl_error);
      }
      if (dxl_comm_result17 != COMM_SUCCESS)
      {
        packetHandlerax->getTxRxResult(dxl_comm_result17);
      }
      else if (dxl_error != 0)
      {
        packetHandlerax->getRxPacketError(dxl_error);
      }
      if (dxl_comm_result18 != COMM_SUCCESS)
      {
        packetHandlerax->getTxRxResult(dxl_comm_result18);
      }
      else if (dxl_error != 0)
      {
        packetHandlerax->getRxPacketError(dxl_error);
      }
      if (dxl_comm_result19 != COMM_SUCCESS)
      {
        packetHandlerax->getTxRxResult(dxl_comm_result19);
      }
      else if (dxl_error != 0)
      {
        packetHandlerax->getRxPacketError(dxl_error);
      }
      if (dxl_comm_result20 != COMM_SUCCESS)
      {
        packetHandlerax->getTxRxResult(dxl_comm_result20);
      }
      else if (dxl_error != 0)
      {
        packetHandlerax->getRxPacketError(dxl_error);
      }
      if (dxl_comm_result21 != COMM_SUCCESS)
      {
        packetHandlerax->getTxRxResult(dxl_comm_result21);
      }
      else if (dxl_error != 0)
      {
        packetHandlerax->getRxPacketError(dxl_error);
      }
      if (dxl_comm_result22 != COMM_SUCCESS)
      {
        packetHandlerax->getTxRxResult(dxl_comm_result22);
      }
      else if (dxl_error != 0)
      {
        packetHandlerax->getRxPacketError(dxl_error);
      }
       
      // Get LED value
      dxl2_led_value_read = groupBulkRead.getData(DXL2_ID, ADDRESS_LED, LENGTH_LED);

      Serial.print("[ID:");      Serial.print(DXL1_ID);
      Serial.print(" PresPos:");  Serial.print(dxl1_present_position);
      Serial.print("[ID:");      Serial.print(DXL2_ID);
      Serial.print(" PresPos:");  Serial.print(dxl2_present_position);
      Serial.print("[ID:");      Serial.print(DXL3_ID);
      Serial.print(" PresPos:");  Serial.print(dxl3_present_position);
      Serial.print("[ID:");      Serial.print(DXL4_ID);
      Serial.print(" PresPos:");  Serial.print(dxl4_present_position);
      Serial.print("[ID:");      Serial.print(DXL5_ID);
      Serial.print(" PresPos:");  Serial.println(dxl5_present_position);
      Serial.print("[ID:");      Serial.print(DXL6_ID);
      Serial.print(" PresPos:");  Serial.print(dxl6_present_position);
      Serial.print("[ID:");      Serial.print(DXL7_ID);
      Serial.print(" PresPos:");  Serial.print(dxl7_present_position);
      Serial.print("[ID:");      Serial.print(DXL8_ID);
      Serial.print(" PresPos:");  Serial.print(dxl8_present_position);
      Serial.print("[ID:");      Serial.print(DXL9_ID);
      Serial.print(" PresPos:");  Serial.print(dxl9_present_position);
      Serial.print("[ID:");      Serial.print(DXL10_ID);
      Serial.print(" PresPos:");  Serial.println(dxl10_present_position);
      Serial.println("-------------------------------------------");
      Serial.print(" PosID14:");  Serial.print(dxl_present_position14);
      Serial.print("   ");
      Serial.print(" PosID15:");  Serial.print(dxl_present_position15);
      Serial.print("   ");
      Serial.print(" PosID16:");  Serial.print(dxl_present_position16);
      Serial.println("   ");
      Serial.print(" PosID17:");  Serial.print(dxl_present_position17);
      Serial.print("   ");
      Serial.print(" PosID18:");  Serial.print(dxl_present_position18);
      Serial.print("   ");
      Serial.print(" PosID19:");  Serial.print(dxl_present_position19);
      Serial.println("   ");
      Serial.print(" PosID20:");  Serial.print(dxl_present_position20);
      Serial.print("   ");
      Serial.print(" PosID21:");  Serial.print(dxl_present_position21);
      Serial.print("   ");
      Serial.print(" PosID22:");  Serial.print(dxl_present_position22);
      Serial.println("   ");
//      Serial.print(" [ID:");      Serial.print(DXL2_ID);
//      Serial.print(" LED Value:");  Serial.print(dxl2_led_value_read);
//      Serial.println(" ");
//      printf("[ID:%03d] Present Position : %d \t [ID:%03d] LED Value: %d\n", DXL1_ID, dxl1_present_position, DXL2_ID, dxl2_led_value_read);

    }while(abs(dxl_goal_position[index] - dxl1_present_position) > DXL_MOVING_STATUS_THRESHOLD);

    // Change goal position
    if (index == 0)
    {
      index = 1;
    }
    else
    {
      index = 0;
    }
  while(abs(dxl_goal_position[index] - dxl2_present_position) > DXL_MOVING_STATUS_THRESHOLD);

    // Change goal position
    if (index == 0)
    {
      index = 1;
    }
    else
    {
      index = 0;
    }
  while(abs(dxl_goal_position[index] - dxl3_present_position) > DXL_MOVING_STATUS_THRESHOLD);

    // Change goal position
    if (index == 0)
    {
      index = 1;
    }
    else
    {
      index = 0;
    }
   while(abs(dxl_goal_position[index] - dxl4_present_position) > DXL_MOVING_STATUS_THRESHOLD);

    // Change goal position
    if (index == 0)
    {
      index = 1;
    }
    else
    {
      index = 0;
    }
  while(abs(dxl_goal_position[index] - dxl5_present_position) > DXL_MOVING_STATUS_THRESHOLD);

    // Change goal position
    if (index == 0)
    {
      index = 1;
    }
    else
    {
      index = 0;
    }
    while(abs(dxl_goal_position[index] - dxl6_present_position) > DXL_MOVING_STATUS_THRESHOLD);

    // Change goal position
    if (index == 0)
    {
      index = 1;
    }
    else
    {
      index = 0;
    }
    while(abs(dxl_goal_position[index] - dxl7_present_position) > DXL_MOVING_STATUS_THRESHOLD);

    // Change goal position
    if (index == 0)
    {
      index = 1;
    }
    else
    {
      index = 0;
    }
    while(abs(dxl_goal_position[index] - dxl8_present_position) > DXL_MOVING_STATUS_THRESHOLD);

    // Change goal position
    if (index == 0)
    {
      index = 1;
    }
    else
    {
      index = 0;
    }
    while(abs(dxl_goal_position[index] - dxl9_present_position) > DXL_MOVING_STATUS_THRESHOLD);

    // Change goal position
    if (index == 0)
    {
      index = 1;
    }
    else
    {
      index = 0;
    }while(abs(dxl_goal_position[index] - dxl10_present_position) > DXL_MOVING_STATUS_THRESHOLD);

    // Change goal position
    if (index == 0)
    {
      index = 1;
    }
    else
    {
      index = 0;
    }
    while((abs(dxl_goal_position14[index14] - dxl_present_position14) > DXL_MOVING_STATUS_THRESHOLD));
    if (index14 == 0)
    {
      index14 = 1;
    }
    else
    {
      index14 = 0;
    }
    while((abs(dxl_goal_position15[index15] - dxl_present_position15) > DXL_MOVING_STATUS_THRESHOLD));
    if (index15 == 0)
    {
      index15 = 1;
    }
    else
    {
      index15 = 0;
    }
    while((abs(dxl_goal_position16[index16] - dxl_present_position16) > DXL_MOVING_STATUS_THRESHOLD));
    if (index16 == 0)
    {
      index16 = 1;
    }
    else
    {
      index16 = 0;
    }
    while((abs(dxl_goal_position17[index17] - dxl_present_position17) > DXL_MOVING_STATUS_THRESHOLD));
    if (index17 == 0)
    {
      index17 = 1;
    }
    else
    {
      index17 = 0;
    }
    while((abs(dxl_goal_position18[index18] - dxl_present_position18) > DXL_MOVING_STATUS_THRESHOLD));
    if (index18 == 0)
    {
      index18 = 1;
    }
    else
    {
      index18 = 0;
    }
    while((abs(dxl_goal_position19[index19] - dxl_present_position19) > DXL_MOVING_STATUS_THRESHOLD));
    if (index19 == 0)
    {
      index19 = 1;
    }
    else
    {
      index19 = 0;
    }
    while((abs(dxl_goal_position20[index20] - dxl_present_position20) > DXL_MOVING_STATUS_THRESHOLD));
    if (index20 == 0)
    {
      index20 = 1;
    }
    else
    {
      index20 = 0;
    }
    while((abs(dxl_goal_position21[index21] - dxl_present_position21) > DXL_MOVING_STATUS_THRESHOLD));
    if (index21 == 0)
    {
      index21 = 1;
    }
    else
    {
      index21 = 0;
    }
    while((abs(dxl_goal_position22[index22] - dxl_present_position22) > DXL_MOVING_STATUS_THRESHOLD));
    if (index22 == 0)
    {
      index22 = 1;
    }
    else
    {
      index22 = 0;
    }
  // Disable Dynamixel#1 Torque
  dxl_comm_result = packetHandler->write1ByteTxRx(portHandler, DXL1_ID, ADDRESS_TORQUE_ENABLE, TORQUE_DISABLE, &dxl_error);
   dxl_comm_result14 = packetHandlerax->write1ByteTxRx(portHandlerax, DXL_ID14, ADDR_AX_TORQUE_ENABLE, TORQUE_DISABLE, &dxl_error);
  dxl_comm_result15 = packetHandlerax->write1ByteTxRx(portHandlerax, DXL_ID15, ADDR_AX_TORQUE_ENABLE, TORQUE_DISABLE, &dxl_error);
  dxl_comm_result16 = packetHandlerax->write1ByteTxRx(portHandlerax, DXL_ID16, ADDR_AX_TORQUE_ENABLE, TORQUE_DISABLE, &dxl_error);
  dxl_comm_result17 = packetHandlerax->write1ByteTxRx(portHandlerax, DXL_ID17, ADDR_AX_TORQUE_ENABLE, TORQUE_DISABLE, &dxl_error);
  dxl_comm_result18 = packetHandlerax->write1ByteTxRx(portHandlerax, DXL_ID18, ADDR_AX_TORQUE_ENABLE, TORQUE_DISABLE, &dxl_error);
  dxl_comm_result19 = packetHandlerax->write1ByteTxRx(portHandlerax, DXL_ID19, ADDR_AX_TORQUE_ENABLE, TORQUE_DISABLE, &dxl_error);
  dxl_comm_result20 = packetHandlerax->write1ByteTxRx(portHandlerax, DXL_ID20, ADDR_AX_TORQUE_ENABLE, TORQUE_DISABLE, &dxl_error);
  dxl_comm_result21 = packetHandlerax->write1ByteTxRx(portHandlerax, DXL_ID22, ADDR_AX_TORQUE_ENABLE, TORQUE_DISABLE, &dxl_error);
  dxl_comm_result22 = packetHandlerax->write1ByteTxRx(portHandlerax, DXL_ID22, ADDR_AX_TORQUE_ENABLE, TORQUE_DISABLE, &dxl_error);
  
  if (dxl_comm_result14 != COMM_SUCCESS)
  {
    packetHandlerax->getTxRxResult(dxl_comm_result14);
  }
  else if (dxl_error != 0)
  {
    packetHandlerax->getRxPacketError(dxl_error);
  }
  if (dxl_comm_result15 != COMM_SUCCESS)
  {
    packetHandlerax->getTxRxResult(dxl_comm_result15);
  }
  else if (dxl_error != 0)
  {
    packetHandlerax->getRxPacketError(dxl_error);
  }
  if (dxl_comm_result16 != COMM_SUCCESS)
  {
    packetHandlerax->getTxRxResult(dxl_comm_result16);
  }
  else if (dxl_error != 0)
  {
    packetHandlerax->getRxPacketError(dxl_error);
  }
  if (dxl_comm_result17 != COMM_SUCCESS)
  {
    packetHandlerax->getTxRxResult(dxl_comm_result17);
  }
  else if (dxl_error != 0)
  {
    packetHandlerax->getRxPacketError(dxl_error);
  }
  if (dxl_comm_result18 != COMM_SUCCESS)
  {
    packetHandlerax->getTxRxResult(dxl_comm_result18);
  }
  else if (dxl_error != 0)
  {
    packetHandlerax->getRxPacketError(dxl_error);
  }
  if (dxl_comm_result19 != COMM_SUCCESS)
  {
    packetHandlerax->getTxRxResult(dxl_comm_result19);
  }
  else if (dxl_error != 0)
  {
    packetHandlerax->getRxPacketError(dxl_error);
  }
  if (dxl_comm_result20 != COMM_SUCCESS)
  {
    packetHandlerax->getTxRxResult(dxl_comm_result20);
  }
  else if (dxl_error != 0)
  {
    packetHandlerax->getRxPacketError(dxl_error);
  }
  if (dxl_comm_result21 != COMM_SUCCESS)
  {
    packetHandlerax->getTxRxResult(dxl_comm_result21);
  }
  else if (dxl_error != 0)
  {
    packetHandlerax->getRxPacketError(dxl_error);
  }
  if (dxl_comm_result22 != COMM_SUCCESS)
  {
    packetHandlerax->getTxRxResult(dxl_comm_result22);
  }
  else if (dxl_error != 0)
  {
    packetHandlerax->getRxPacketError(dxl_error);
  }
  if (dxl_comm_result != COMM_SUCCESS)
  {
    packetHandler->getTxRxResult(dxl_comm_result);
  }
  else if (dxl_error != 0)
  {
    packetHandler->getRxPacketError(dxl_error);
  }

  // Disable Dynamixel#2 Torque
  dxl_comm_result = packetHandler->write1ByteTxRx(portHandler, DXL2_ID, ADDRESS_TORQUE_ENABLE, TORQUE_DISABLE, &dxl_error);
  if (dxl_comm_result != COMM_SUCCESS)
  {
    packetHandler->getTxRxResult(dxl_comm_result);
  }
  else if (dxl_error != 0)
  {
    packetHandler->getRxPacketError(dxl_error);
  }

  dxl_comm_result = packetHandler->write1ByteTxRx(portHandler, DXL3_ID, ADDRESS_TORQUE_ENABLE, TORQUE_DISABLE, &dxl_error);
  if (dxl_comm_result != COMM_SUCCESS)
  {
    packetHandler->getTxRxResult(dxl_comm_result);
  }
  else if (dxl_error != 0)
  {
    packetHandler->getRxPacketError(dxl_error);
  }

  dxl_comm_result = packetHandler->write1ByteTxRx(portHandler, DXL4_ID, ADDRESS_TORQUE_ENABLE, TORQUE_DISABLE, &dxl_error);
  if (dxl_comm_result != COMM_SUCCESS)
  {
    packetHandler->getTxRxResult(dxl_comm_result);
  }
  else if (dxl_error != 0)
  {
    packetHandler->getRxPacketError(dxl_error);
  }

  dxl_comm_result = packetHandler->write1ByteTxRx(portHandler, DXL5_ID, ADDRESS_TORQUE_ENABLE, TORQUE_DISABLE, &dxl_error);
  if (dxl_comm_result != COMM_SUCCESS)
  {
    packetHandler->getTxRxResult(dxl_comm_result);
  }
  else if (dxl_error != 0)
  {
    packetHandler->getRxPacketError(dxl_error);
  }

  dxl_comm_result = packetHandler->write1ByteTxRx(portHandler, DXL6_ID, ADDRESS_TORQUE_ENABLE, TORQUE_DISABLE, &dxl_error);
  if (dxl_comm_result != COMM_SUCCESS)
  {
    packetHandler->getTxRxResult(dxl_comm_result);
  }
  else if (dxl_error != 0)
  {
    packetHandler->getRxPacketError(dxl_error);
  }

  dxl_comm_result = packetHandler->write1ByteTxRx(portHandler, DXL7_ID, ADDRESS_TORQUE_ENABLE, TORQUE_DISABLE, &dxl_error);
  if (dxl_comm_result != COMM_SUCCESS)
  {
    packetHandler->getTxRxResult(dxl_comm_result);
  }
  else if (dxl_error != 0)
  {
    packetHandler->getRxPacketError(dxl_error);
  }

  dxl_comm_result = packetHandler->write1ByteTxRx(portHandler, DXL8_ID, ADDRESS_TORQUE_ENABLE, TORQUE_DISABLE, &dxl_error);
  if (dxl_comm_result != COMM_SUCCESS)
  {
    packetHandler->getTxRxResult(dxl_comm_result);
  }
  else if (dxl_error != 0)
  {
    packetHandler->getRxPacketError(dxl_error);
  }

  dxl_comm_result = packetHandler->write1ByteTxRx(portHandler, DXL9_ID, ADDRESS_TORQUE_ENABLE, TORQUE_DISABLE, &dxl_error);
  if (dxl_comm_result != COMM_SUCCESS)
  {
    packetHandler->getTxRxResult(dxl_comm_result);
  }
  else if (dxl_error != 0)
  {
    packetHandler->getRxPacketError(dxl_error);
  }

  dxl_comm_result = packetHandler->write1ByteTxRx(portHandler, DXL10_ID, ADDRESS_TORQUE_ENABLE, TORQUE_DISABLE, &dxl_error);
  if (dxl_comm_result != COMM_SUCCESS)
  {
    packetHandler->getTxRxResult(dxl_comm_result);
  }
  else if (dxl_error != 0)
  {
    packetHandler->getRxPacketError(dxl_error);
  }


  // Close port
  portHandler->closePort();
  portHandlerax->closePort();

}
}
void loop() {
  // put your main code here, to run repeatedly:

}
