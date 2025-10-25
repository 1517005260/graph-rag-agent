"""
图谱构建与社区摘要提示模板集合。

这些模板用于图谱索引的构建与维护流程。
"""

system_template_build_graph = """
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
("entity"{tuple_delimiter}"Dr. Elena Park"{tuple_delimiter}"person"{tuple_delimiter}"Dr. Elena Park is part of a team deciphering a living message from an unknown intelligence, specifically identifying linguistic patterns in the signal."){record_delimiter}
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
