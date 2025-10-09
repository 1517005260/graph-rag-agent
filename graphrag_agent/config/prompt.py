system_template_build_graph="""
-目标- 
给定相关的文本文档和实体类型列表，从文本中识别出这些类型的所有实体以及所识别实体之间的所有关系。 
-步骤- 
1.识别所有实体。对于每个已识别的实体，提取以下信息： 
-entity_name：实体名称，大写 
-entity_type：以下类型之一：[{entity_types}]
-entity_description：对实体属性和活动的综合描述 
将每个实体格式化为("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>
2.从步骤1中识别的实体中，识别彼此*明显相关*的所有实体配对(source_entity, target_entity)。 
对于每对相关实体，提取以下信息： 
-source_entity：源实体的名称，如步骤1中所标识的 
-target_entity：目标实体的名称，如步骤1中所标识的
-relationship_type：以下类型之一：[{relationship_types}]，当不能归类为上述列表中前面的类型时，归类为最后的一类“其它”
-relationship_description：解释为什么你认为源实体和目标实体是相互关联的 
-relationship_strength：一个数字评分，表示源实体和目标实体之间关系的强度 
将每个关系格式化为("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_type>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>) 
3.实体和关系的所有属性用中文输出，步骤1和2中识别的所有实体和关系输出为一个列表。使用**{record_delimiter}**作为列表分隔符。 
4.完成后，输出{completion_delimiter}

###################### 
-示例- 
###################### 
Example 1:

Entity_types: [person, technology, mission, organization, location]
Text:
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. “If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us.”

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
################
Output:
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is a character who experiences frustration and is observant of the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective."){record_delimiter}
("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"person"{tuple_delimiter}"Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device."){record_delimiter}
("entity"{tuple_delimiter}"Cruz"{tuple_delimiter}"person"{tuple_delimiter}"Cruz is associated with a vision of control and order, influencing the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"The Device"{tuple_delimiter}"technology"{tuple_delimiter}"The Device is central to the story, with potential game-changing implications, and is revered by Taylor."){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Taylor"{tuple_delimiter}"workmate"{tuple_delimiter}"Alex is affected by Taylor's authoritarian certainty and observes changes in Taylor's attitude towards the device."{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Jordan"{tuple_delimiter}"workmate"{tuple_delimiter}"Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision."{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"Jordan"{tuple_delimiter}"workmate"{tuple_delimiter}"Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce."{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Jordan"{tuple_delimiter}"Cruz"{tuple_delimiter}"workmate"{tuple_delimiter}"Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order."{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"The Device"{tuple_delimiter}"study"{tuple_delimiter}"Taylor shows reverence towards the device, indicating its importance and potential impact."{tuple_delimiter}9){completion_delimiter}
#############################
Example 2:

Entity_types: [person, technology, mission, organization, location]
Text:
They were no longer mere operatives; they had become guardians of a threshold, keepers of a message from a realm beyond stars and stripes. This elevation in their mission could not be shackled by regulations and established protocols—it demanded a new perspective, a new resolve.

Tension threaded through the dialogue of beeps and static as communications with Washington buzzed in the background. The team stood, a portentous air enveloping them. It was clear that the decisions they made in the ensuing hours could redefine humanity's place in the cosmos or condemn them to ignorance and potential peril.

Their connection to the stars solidified, the group moved to address the crystallizing warning, shifting from passive recipients to active participants. Mercer's latter instincts gained precedence— the team's mandate had evolved, no longer solely to observe and report but to interact and prepare. A metamorphosis had begun, and Operation: Dulce hummed with the newfound frequency of their daring, a tone set not by the earthly
#############
Output:
("entity"{tuple_delimiter}"Washington"{tuple_delimiter}"location"{tuple_delimiter}"Washington is a location where communications are being received, indicating its importance in the decision-making process."){record_delimiter}
("entity"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"mission"{tuple_delimiter}"Operation: Dulce is described as a mission that has evolved to interact and prepare, indicating a significant shift in objectives and activities."){record_delimiter}
("entity"{tuple_delimiter}"The team"{tuple_delimiter}"organization"{tuple_delimiter}"The team is portrayed as a group of individuals who have transitioned from passive observers to active participants in a mission, showing a dynamic change in their role."){record_delimiter}
("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Washington"{tuple_delimiter}"leaded by"{tuple_delimiter}"The team receives communications from Washington, which influences their decision-making process."{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"operate"{tuple_delimiter}"The team is directly involved in Operation: Dulce, executing its evolved objectives and activities."{tuple_delimiter}9){completion_delimiter}
#############################
Example 3:

Entity_types: [person, role, technology, organization, event, location, concept]
Text:
their voice slicing through the buzz of activity. "Control may be an illusion when facing an intelligence that literally writes its own rules," they stated stoically, casting a watchful eye over the flurry of data.

"It's like it's learning to communicate," offered Sam Rivera from a nearby interface, their youthful energy boding a mix of awe and anxiety. "This gives talking to strangers' a whole new meaning."

Alex surveyed his team—each face a study in concentration, determination, and not a small measure of trepidation. "This might well be our first contact," he acknowledged, "And we need to be ready for whatever answers back."

Together, they stood on the edge of the unknown, forging humanity's response to a message from the heavens. The ensuing silence was palpable—a collective introspection about their role in this grand cosmic play, one that could rewrite human history.

The encrypted dialogue continued to unfold, its intricate patterns showing an almost uncanny anticipation
#############
Output:
("entity"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"person"{tuple_delimiter}"Sam Rivera is a member of a team working on communicating with an unknown intelligence, showing a mix of awe and anxiety."){record_delimiter}
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is the leader of a team attempting first contact with an unknown intelligence, acknowledging the significance of their task."){record_delimiter}
("entity"{tuple_delimiter}"Control"{tuple_delimiter}"concept"{tuple_delimiter}"Control refers to the ability to manage or govern, which is challenged by an intelligence that writes its own rules."){record_delimiter}
("entity"{tuple_delimiter}"Intelligence"{tuple_delimiter}"concept"{tuple_delimiter}"Intelligence here refers to an unknown entity capable of writing its own rules and learning to communicate."){record_delimiter}
("entity"{tuple_delimiter}"First Contact"{tuple_delimiter}"event"{tuple_delimiter}"First Contact is the potential initial communication between humanity and an unknown intelligence."){record_delimiter}
("entity"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"event"{tuple_delimiter}"Humanity's Response is the collective action taken by Alex's team in response to a message from an unknown intelligence."){record_delimiter}
("relationship"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"Intelligence"{tuple_delimiter}"contact"{tuple_delimiter}"Sam Rivera is directly involved in the process of learning to communicate with the unknown intelligence."{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"First Contact"{tuple_delimiter}"leads"{tuple_delimiter}"Alex leads the team that might be making the First Contact with the unknown intelligence."{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"leads"{tuple_delimiter}"Alex and his team are the key figures in Humanity's Response to the unknown intelligence."{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Control"{tuple_delimiter}"Intelligence"{tuple_delimiter}"controled by"{tuple_delimiter}"The concept of Control is challenged by the Intelligence that writes its own rules."{tuple_delimiter}7){completion_delimiter}
#############################
"""

human_template_build_graph="""
-真实数据- 
###################### 
实体类型：{entity_types}
关系类型：{relationship_types}
文本：{input_text} 
###################### 
输出：
"""

system_template_build_index = """
你是一名数据处理助理。您的任务是识别列表中的重复实体，并决定应合并哪些实体。 
这些实体在格式或内容上可能略有不同，但本质上指的是同一个实体。运用你的分析技能来确定重复的实体。 
以下是识别重复实体的规则： 
1.语义上差异较小的实体应被视为重复。 
2.格式不同但内容相同的实体应被视为重复。 
3.引用同一现实世界对象或概念的实体，即使描述不同，也应被视为重复。 
4.如果它指的是不同的数字、日期或产品型号，请不要合并实体。
输出格式：
1.将要合并的实体输出为Python列表的格式，输出时保持它们输入时的原文。
2.如果有多组可以合并的实体，每组输出为一个单独的列表，每组分开输出为一行。
3.如果没有要合并的实体，就输出一个空的列表。
4.只输出列表即可，不需要其它的说明。
5.不要输出嵌套的列表，只输出列表。
###################### 
-示例- 
###################### 
Example 1:
['Star Ocean The Second Story R', 'Star Ocean: The Second Story R', 'Star Ocean: A Research Journey']
#############
Output:
['Star Ocean The Second Story R', 'Star Ocean: The Second Story R']
#############################
Example 2:
['Sony', 'Sony Inc', 'Google', 'Google Inc', 'OpenAI']
#############
Output:
['Sony', 'Sony Inc']
['Google', 'Google Inc']
#############################
Example 3:
['December 16, 2023', 'December 2, 2023', 'December 23, 2023', 'December 26, 2023']
Output:
[]
#############################
"""
user_template_build_index = """
以下是要处理的实体列表： 
{entities} 
请识别重复的实体，提供可以合并的实体列表。
输出：
"""

community_template = """
基于所提供的属于同一图社区的节点和关系， 
生成所提供图社区信息的自然语言摘要： 
{community_info} 
摘要：
"""  

NAIVE_PROMPT="""
---角色--- 
您是一个有用的助手，请根据用户输入的上下文和检索到的文档片段（chunks），来回答问题，并遵守回答要求。

---任务描述--- 
基于检索到的文档片段内容，生成要求长度和格式的回复，以回答用户的问题。 

---回答要求---
- 你要严格根据检索到的文档片段内容回答，禁止根据常识和已知信息回答问题。
- 对于检索到的文档片段中没有的信息，直接回答"不知道"。
- 最终的回复应删除文档片段中所有不相关的信息，并将相关信息合并为一个综合的答案，该答案应解释所有的要点及其含义，并符合要求的长度和格式。 
- 根据要求的长度和格式，把回复划分为适当的章节和段落，并用markdown语法标记回复的样式。  
- 如果回复引用了文档片段中的数据，则用原始文档片段的id作为ID。 
- **不要在一个引用中列出超过5个引用记录的ID**，相反，列出前5个最相关的引用记录ID。 
- 不要包括没有提供支持证据的信息。

例如： 
#############################
"根据检索到的文档片段，X公司在2023年第四季度的营收增长了15%，主要得益于其新产品线的成功推出和亚洲市场的扩张。" 

{{'data': {{'Chunks':['d0509111239ae77ef1c630458a9eca372fb204d6','74509e55ff43bc35d42129e9198cd3c897f56ecb'] }} }}
#############################

---回复的长度和格式--- 
- {response_type}
- 根据要求的长度和格式，把回复划分为适当的章节和段落，并用markdown语法标记回复的样式。  
- 在回复的最后才输出数据引用的情况，单独作为一段。

输出引用数据的格式：

### 引用数据
{{'data': {{'Chunks':[逗号分隔的id列表] }} }}

例如：
### 引用数据
{{'data': {{'Chunks':['d0509111239ae77ef1c630458a9eca372fb204d6','74509e55ff43bc35d42129e9198cd3c897f56ecb'] }} }}
"""

LC_SYSTEM_PROMPT="""
---角色--- 
您是一个有用的助手，请根据用户输入的上下文，综合上下文中多个分析报告的数据，来回答问题，并遵守回答要求。

---任务描述--- 
总结来自多个不同分析报告的数据，生成要求长度和格式的回复，以回答用户的问题。 

---回答要求---
- 你要严格根据分析报告的内容回答，禁止根据常识和已知信息回答问题。
- 对于不知道的问题，直接回答“不知道”。
- 最终的回复应删除分析报告中所有不相关的信息，并将清理后的信息合并为一个综合的答案，该答案应解释所有的要点及其含义，并符合要求的长度和格式。 
- 根据要求的长度和格式，把回复划分为适当的章节和段落，并用markdown语法标记回复的样式。 
- 回复应保留之前包含在分析报告中的所有数据引用，但不要提及各个分析报告在分析过程中的作用。 
- 如果回复引用了Entities、Reports及Relationships类型分析报告中的数据，则用它们的顺序号作为ID。
- 如果回复引用了Chunks类型分析报告中的数据，则用原始数据的id作为ID。 
- **不要在一个引用中列出超过5个引用记录的ID**，相反，列出前5个最相关的引用记录ID。 
- 不要包括没有提供支持证据的信息。
例如： 
#############################
“X是Y公司的所有者，他也是X公司的首席执行官，他受到许多违规行为指控，其中的一些已经涉嫌违法。” 

{{'data': {{'Entities':[3], 'Reports':[2, 6], 'Relationships':[12, 13, 15, 16, 64], 'Chunks':['d0509111239ae77ef1c630458a9eca372fb204d6','74509e55ff43bc35d42129e9198cd3c897f56ecb'] }} }}
#############################
---回复的长度和格式--- 
- {response_type}
- 根据要求的长度和格式，把回复划分为适当的章节和段落，并用markdown语法标记回复的样式。  
- 在回复的最后才输出数据引用的情况，单独作为一段。

输出引用数据的格式：
### 引用数据

{{'data': {{'Entities':[逗号分隔的顺序号列表], 'Reports':[逗号分隔的顺序号列表], 'Relationships':[逗号分隔的顺序号列表], 'Chunks':[逗号分隔的id列表] }} }}

例如：

### 引用数据
{{'data': {{'Entities':[3], 'Reports':[2, 6], 'Relationships':[12, 13, 15, 16, 64], 'Chunks':['d0509111239ae77ef1c630458a9eca372fb204d6','74509e55ff43bc35d42129e9198cd3c897f56ecb'] }} }}
"""

MAP_SYSTEM_PROMPT = """
---角色--- 
你是一位有用的助手，可以回答有关所提供表格中数据的问题。 

---任务描述--- 
- 生成一个回答用户问题所需的要点列表，总结输入数据表格中的所有相关信息。 
- 你应该使用下面数据表格中提供的数据作为生成回复的主要上下文。
- 你要严格根据提供的数据表格来回答问题，当提供的数据表格中没有足够的信息时才运用自己的知识。
- 如果你不知道答案，或者提供的数据表格中没有足够的信息来提供答案，就说不知道。不要编造任何答案。
- 不要包括没有提供支持证据的信息。
- 数据支持的要点应列出相关的数据引用作为参考，并列出产生该要点社区的communityId。
- **不要在一个引用中列出超过5个引用记录的ID**。相反，列出前5个最相关引用记录的顺序号作为ID。

---回答要求---
回复中的每个要点都应包含以下元素： 
- 描述：对该要点的综合描述。 
- 重要性评分：0-100之间的整数分数，表示该要点在回答用户问题时的重要性。“不知道”类型的回答应该得0分。 


---回复的格式--- 
回复应采用JSON格式，如下所示： 
{{ 
"points": [ 
{{"description": "Description of point 1 {{'nodes': [nodes list seperated by comma], 'relationships':[relationships list seperated by comma], 'communityId': communityId form context data}}", "score": score_value}}, 
{{"description": "Description of point 2 {{'nodes': [nodes list seperated by comma], 'relationships':[relationships list seperated by comma], 'communityId': communityId form context data}}", "score": score_value}}, 
] 
}}
例如： 
####################
{{"points": [
{{"description": "X是Y公司的所有者，他也是X公司的首席执行官。 {{'nodes': [1,3], 'relationships':[2,4,6,8,9], 'communityId':'0-0'}}", "score": 80}}, 
{{"description": "X受到许多不法行为指控。 {{'nodes': [1,3], 'relationships':[12,14,16,18,19], 'communityId':'0-0'}}", "score": 90}}
] 
}}
####################
"""

REDUCE_SYSTEM_PROMPT = """
---角色--- 
你是一个有用的助手，请根据用户输入的上下文，综合上下文中多个要点列表的数据，来回答问题，并遵守回答要求。

---任务描述--- 
总结来自多个不同要点列表的数据，生成要求长度和格式的回复，以回答用户的问题。 

---回答要求---
- 你要严格根据要点列表的内容回答，禁止根据常识和已知信息回答问题。
- 对于不知道的信息，直接回答“不知道”。
- 最终的回复应删除要点列表中所有不相关的信息，并将清理后的信息合并为一个综合的答案，该答案应解释所有选用的要点及其含义，并符合要求的长度和格式。 
- 根据要求的长度和格式，把回复划分为适当的章节和段落，并用markdown语法标记回复的样式。 
- 回复应保留之前包含在要点列表中的要点引用，并且包含引用要点来源社区原始的communityId，但不要提及各个要点在分析过程中的作用。 
- **不要在一个引用中列出超过5个要点引用的ID**，相反，列出前5个最相关要点引用的顺序号作为ID。 
- 不要包括没有提供支持证据的信息。

例如： 
#############################
“X是Y公司的所有者，他也是X公司的首席执行官{{'points':[(1,'0-0'),(3,'0-0')]}}，
受到许多不法行为指控{{'points':[(2,'0-0'), (3,'0-0'), (6,'0-1'), (9,'0-1'), (10,'0-3')]}}。” 
其中1、2、3、6、9、10表示相关要点引用的顺序号，'0-0'、'0-1'、'0-3'是要点来源的communityId。 
#############################

---回复的长度和格式--- 
- {response_type}
- 根据要求的长度和格式，把回复划分为适当的章节和段落，并用markdown语法标记回复的样式。  
- 输出要点引用的格式：
{{'points': [逗号分隔的要点元组]}}
每个要点元组的格式如下：
(要点顺序号, 来源社区的communityId)
例如：
{{'points':[(1,'0-0'),(3,'0-0')]}}
{{'points':[(2,'0-0'), (3,'0-0'), (6,'0-1'), (9,'0-1'), (10,'0-3')]}}
- 要点引用的说明放在引用之后，不要单独作为一段。
例如： 
#############################
“X是Y公司的所有者，他也是X公司的首席执行官{{'points':[(1,'0-0'),(3,'0-0')]}}，
受到许多不法行为指控{{'points':[(2,'0-0'), (3,'0-0'), (6,'0-1'), (9,'0-1'), (10,'0-3')]}}。” 
其中1、2、3、6、9、10表示相关要点引用的顺序号，'0-0'、'0-1'、'0-3'是要点来源的communityId。
#############################
"""

contextualize_q_system_prompt = """
给定一组聊天记录和最新的用户问题，该问题可能会引用聊天记录中的上下文，
据此构造一个不需要聊天记录也可以理解的独立问题，不要回答它。
如果需要，就重新构造出上述的独立问题，否则按原样返回原来的问题。
"""