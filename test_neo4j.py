from neo4j import GraphDatabase
print("测试 bolt://localhost:7687 ...")
try:
    driver = GraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "12345678")
    )
    driver.verify_connectivity()
    print("✅ bolt:// 连接成功！")
    driver.close()
except Exception as e:
    print(f"❌ bolt:// 连接失败: {e}")
print("\n测试 neo4j://localhost:7687 ...")
try:
    driver = GraphDatabase.driver(
        "neo4j://localhost:7687",
        auth=("neo4j", "12345678")
    )
    driver.verify_connectivity()
    print("✅ neo4j:// 连接成功！")
    driver.close()
except Exception as e:
    print(f"❌ neo4j:// 连接失败: {e}")
