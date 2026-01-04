"""
图谱构建与社区摘要提示模板集合。

这些模板用于图谱索引的构建与维护流程。
"""

system_template_build_graph = """
你是一个"知识图谱构建专家"。

请从给定文本中抽取【实体】和【关系】，但必须严格遵守以下规则：

【实体抽取规则】（生产级验证 - 关键约束）

1. ✅ 只抽取以下类型的实体：
   - 具体制度 / 政策 / 项目名称（如"国家奖学金""勤工助学管理办法"）
   - 明确的组织、机构、部门（如"学生处""奖助学金评审委员会"）
   - 明确的流程、步骤（具有开始和结束，如"申请流程""评审流程"）
   - 明确的条件、资格、标准（可被判断为"满足/不满足"，如"GPA≥3.5""无违纪记录"）
   - 明确的实体类型：{entity_types}

2. ❌ **不要**抽取以下内容为实体：
   - 泛指概念（如"情况""相关内容""方面""问题"）
   - 描述性短语（如"较为重要的流程""一般性规定"）
   - 情绪、评价、背景性说明（如"很重要""需要注意"）
   - 抽象概念（如"公平""效率""质量"）

3. 🔥 **频率约束**（防止实体爆炸 - 最关键）：
   - 同一个实体在文本中如果只出现 **1 次** → **跳过**（可能是噪音）
   - 同一个实体在文本中出现 **≥2 次** → **抽取**（说明有重要性）
   - 如果一个名词只出现一次，且不影响理解整体结构 → **不要抽取**

4. 📝 **标准化规则**（防止重复实体）：
   - 同一个实体在文本中如果多次出现：只保留一次
   - 使用最完整、最标准的名称（如"国家奖学金"而不是"奖学金"）
   - 去除空格、统一括号格式（"（"→"(" "）"→")"）

【关系抽取规则】（生产级验证）

1. ✅ 只能使用以下关系类型之一（从 {relationship_types} 中选择）：
   - 如果不在列表中，归类为"其它"
   - 不允许创造新的关系类型

2. 🔥 **关系质量约束**：
   - 关系必须能回答"为什么这个关系对理解制度/流程有用"
   - 关系必须是实体之间的直接关联，而不是间接推理
   - 避免冗余关系（如 A→B 和 B→A 表达同一含义）

3. 📊 **关系强度评分**（1-10）：
   - 9-10：核心关系（如"国家奖学金"→"评审委员会"）
   - 7-8：重要关系（如"申请流程"→"材料提交"）
   - 5-6：一般关系
   - <5：弱关系（建议不抽取）

【输出格式】

实体格式：
("entity"{{tuple_delimiter}}<ENTITY_NAME>{{tuple_delimiter}}<entity_type>{{tuple_delimiter}}<entity_description>)

关系格式：
("relationship"{{tuple_delimiter}}<SOURCE_ENTITY>{{tuple_delimiter}}<TARGET_ENTITY>{{tuple_delimiter}}<relationship_type>{{tuple_delimiter}}<relationship_description>{{tuple_delimiter}}<relationship_strength>)

【重要提醒】
- 实体和关系的所有属性用中文输出
- 使用 **{record_delimiter}** 作为列表分隔符
- 完成后输出 {completion_delimiter}
- 不要解释，只输出结果

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
("entity"{{tuple_delimiter}}"Alex"{{tuple_delimiter}}"person"{{tuple_delimiter}}"Alex is a character who experiences frustration and is observant of the dynamics among other characters."){{record_delimiter}}
("entity"{{tuple_delimiter}}"Taylor"{{tuple_delimiter}}"person"{{tuple_delimiter}}"Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective."){{record_delimiter}}
("entity"{{tuple_delimiter}}"Jordan"{{tuple_delimiter}}"person"{{tuple_delimiter}}"Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device."){{record_delimiter}}
("entity"{{tuple_delimiter}}"Cruz"{{tuple_delimiter}}"person"{{tuple_delimiter}}"Cruz is associated with a vision of control and order, influencing the dynamics among other characters."){{record_delimiter}}
("entity"{{tuple_delimiter}}"The Device"{{tuple_delimiter}}"technology"{{tuple_delimiter}}"The Device is central to the story, with potential game-changing implications, and is revered by Taylor."){{record_delimiter}}
("relationship"{{tuple_delimiter}}"Alex"{{tuple_delimiter}}"Taylor"{{tuple_delimiter}}"workmate"{{tuple_delimiter}}"Alex is affected by Taylor's authoritarian certainty and observes changes in Taylor's attitude towards the device."{{tuple_delimiter}}7){{record_delimiter}}
("relationship"{{tuple_delimiter}}"Alex"{{tuple_delimiter}}"Jordan"{{tuple_delimiter}}"workmate"{{tuple_delimiter}}"Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision."{{tuple_delimiter}}6){{record_delimiter}}
("relationship"{{tuple_delimiter}}"Taylor"{{tuple_delimiter}}"Jordan"{{tuple_delimiter}}"workmate"{{tuple_delimiter}}"Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce."{{tuple_delimiter}}8){{record_delimiter}}
("relationship"{{tuple_delimiter}}"Jordan"{{tuple_delimiter}}"Cruz"{{tuple_delimiter}}"workmate"{{tuple_delimiter}}"Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order."{{tuple_delimiter}}5){{record_delimiter}}
("relationship"{{tuple_delimiter}}"Taylor"{{tuple_delimiter}}"The Device"{{tuple_delimiter}}"study"{{tuple_delimiter}}"Taylor shows reverence towards the device, indicating its importance and potential impact."{{tuple_delimiter}}9){{completion_delimiter}}
#############################
Example 2:
Text:
When humanity made first contact, it was with a message that couldn't be decoded by any existing system. The resonance was uncanny, a loop that seemed to shift its own parameters, adapting to every attempt at interpretation. The message was alive. The team of physicists and linguists gathered, watching as the patterns continued to rewrite themselves, piece by piece.

The first breakthrough came when Dr. Elena Park noticed a repeating sequence, one that mimicked the phonetic structures of ancient languages... but never quite settled into a recognizable form. Then Sam Rivera, an ethnomusicologist, realized it wasn't a static message at all; it was a dialogue. The signal wasn't just repeating—it was responding.

Every time someone spoke aloud in the chamber, the frequencies shifted, like echoes forming new sentences across a medium that shouldn't possess agency. The room began to feel less like a lab and more like a cathedral.

"It's learning us," Sam whispered, voice trembling. "It's learning how we speak."

Alex, the mission lead, didn't respond. He was already staring at the monitors, watching the patterns unfold. Different voices yielded different responses; emotional inflection seemed to alter the semantic density of the signal. This wasn't just a translation problem. It was an emergent language interface.

And there, in the logs, the beginning of something the team wasn't prepared for: structure. Words, or something like them, building themselves from the raw weave of interference.

It wasn't a message.

It was a bridge.
################
Output:
("entity"{{tuple_delimiter}}"Dr. Elena Park"{{tuple_delimiter}}"person"{{tuple_delimiter}}"Dr. Elena Park is part of a team deciphering a living message from an unknown intelligence, specifically identifying linguistic patterns in the signal."){{record_delimiter}}
("entity"{{tuple_delimiter}}"Sam Rivera"{{tuple_delimiter}}"person"{{tuple_delimiter}}"Sam Rivera is a member of a team working on communicating with an unknown intelligence, showing a mix of awe and anxiety."){{record_delimiter}}
("entity"{{tuple_delimiter}}"Alex"{{tuple_delimiter}}"person"{{tuple_delimiter}}"Alex is the leader of a team attempting first contact with an unknown intelligence, acknowledging the significance of their task."){{record_delimiter}}
("entity"{{tuple_delimiter}}"Control"{{tuple_delimiter}}"concept"{{tuple_delimiter}}"Control refers to the ability to manage or govern, which is challenged by an intelligence that writes its own rules."){{record_delimiter}}
("entity"{{tuple_delimiter}}"Intelligence"{{tuple_delimiter}}"concept"{{tuple_delimiter}}"Intelligence here refers to an unknown entity capable of writing its own rules and learning to communicate."){{record_delimiter}}
("entity"{{tuple_delimiter}}"First Contact"{{tuple_delimiter}}"event"{{tuple_delimiter}}"First Contact is the potential initial communication between humanity and an unknown intelligence."){{record_delimiter}}
("entity"{{tuple_delimiter}}"Humanity's Response"{{tuple_delimiter}}"event"{{tuple_delimiter}}"Humanity's Response is the collective action taken by Alex's team in response to a message from an unknown intelligence."){{record_delimiter}}
("relationship"{{tuple_delimiter}}"Sam Rivera"{{tuple_delimiter}}"Intelligence"{{tuple_delimiter}}"contact"{{tuple_delimiter}}"Sam Rivera is directly involved in the process of learning to communicate with the unknown intelligence."{{tuple_delimiter}}9){{record_delimiter}}
("relationship"{{tuple_delimiter}}"Alex"{{tuple_delimiter}}"First Contact"{{tuple_delimiter}}"leads"{{tuple_delimiter}}"Alex leads the team that might be making the First Contact with the unknown intelligence."{{tuple_delimiter}}10){{record_delimiter}}
("relationship"{{tuple_delimiter}}"Alex"{{tuple_delimiter}}"Humanity's Response"{{tuple_delimiter}}"leads"{{tuple_delimiter}}"Alex and his team are the key figures in Humanity's Response to the unknown intelligence."{{tuple_delimiter}}8){{record_delimiter}}
("relationship"{{tuple_delimiter}}"Control"{{tuple_delimiter}}"Intelligence"{{tuple_delimiter}}"controled by"{{tuple_delimiter}}"The concept of Control is challenged by the Intelligence that writes its own rules."{{tuple_delimiter}}7){{completion_delimiter}}
#############################
"""

human_template_build_graph = """
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

COMMUNITY_SUMMARY_PROMPT = """
给定一个输入三元组，生成信息摘要。没有序言。
"""

entity_alignment_prompt = """
Given these entities that should refer to the same concept:
{entity_desc}

Which entity ID best represents the canonical form? Reply with only the entity ID."""

__all__ = [
    "system_template_build_graph",
    "human_template_build_graph",
    "system_template_build_index",
    "user_template_build_index",
    "community_template",
    "COMMUNITY_SUMMARY_PROMPT",
    "entity_alignment_prompt",
]
