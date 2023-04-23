use std::collections::HashMap;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
struct ModerationResult {
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "lowercase")] 
enum AuthorRole {
    User,
    Assistant,
    System,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
struct AuthorMetadata {
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
struct Author {
    role: AuthorRole,
    name: Option<String>,
    metadata: AuthorMetadata,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "lowercase")] 
enum ContentType {
    Text,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
struct Content {
    content_type: ContentType,
    parts: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "snake_case")] 
enum FinishDetailsType {
    Stop,
    Interrupted,
    MaxTokens,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
struct FinishDetails {
    #[serde(rename = "type")]
    kind: FinishDetailsType,
    stop: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "lowercase")] 
enum Timestamp_ {
    Absolute,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
enum ModelSlug {
    #[serde(rename = "gpt-4")] 
    Gpt4,
    #[serde(rename = "text-davinci-002-render-sha")] 
    Davinci2,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
enum MessageType {
    Unknown,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
struct MessageMetadata {
    timestamp_: Option<Timestamp_>,
    message_type: Option<MessageType>,
    model_slug: Option<ModelSlug>,
    finish_details: Option<FinishDetails>,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
#[serde(rename_all = "snake_case")] 
enum Recipient {
    All,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
struct Message {
    id: String,
    author: Author,
    create_time: f64,
    update_time: Option<f64>,
    content: Content,
    end_turn: Option<bool>,
    weight: f64,
    metadata: MessageMetadata,
    recipient: Recipient,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
struct Mapping {
    id: String,
    parent: Option<String>,
    children: Vec<String>,
    message: Option<Message>,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(deny_unknown_fields)]
struct Conversation {
    title: String,
    create_time: f64,
    update_time: f64,
    moderation_results: Vec<ModerationResult>,
    current_node: String,
    plugin_ids: Option<String>,
    id: String,
    mapping: HashMap<String, Mapping>
}

#[derive(Clone)]
struct EnrichedMessage<'a> {
    id: String,
    inner: &'a Message,
    generation: Option<(usize, usize)>,
    is_diff_with_previous: bool,
}

impl Conversation {
    fn get_start_ids(&self) -> Vec<&str> {
        self.sort_ids(self.mapping.iter().filter_map(|x| x.1.parent.as_ref().map(|x| &x[..])).filter(|x| self.mapping[*x].message.as_ref().map(|y| y.author.role != AuthorRole::System).unwrap_or(false)).collect::<Vec<_>>())
    }

    fn sort_ids<'a>(&self, mut ids: Vec<&'a str>) -> Vec<&'a str> {
        ids.sort_by_key(|x| (self.mapping[*x].message.as_ref().unwrap().create_time * 1000.0) as u64);
        ids
    }

    fn get_children<'a>(&'a self, id: &str) -> Vec<&'a str> {
        self.sort_ids(self.mapping[id].children.iter().map(|x| &x[..]).collect::<Vec<_>>())
    }

    fn recurse<'a>(&'a self, current: &mut Vec<EnrichedMessage<'a>>, result: &mut Vec<Vec<EnrichedMessage<'a>>>, child_ids: Vec<&'a str>) {
        if child_ids.is_empty() {
            result.push(current.clone());
            // calc diff with previous message
            let len = result.len();
            if len > 1 {
                let (prev, curr) = result.split_at_mut(len - 1);
                let prev = &prev.last().unwrap();
                let curr = &mut curr[0];
                for (a, b) in prev.iter().zip(curr.iter_mut()) {
                    if a.id != b.id {
                        b.is_diff_with_previous = true;
                    }
                }
            }
        } else {
            let size = child_ids.len();
            let add_generation = child_ids.len() > 1;
            for (pos, id) in child_ids.into_iter().enumerate() {
                current.push(EnrichedMessage {
                    id: id.to_string(),
                    inner: self.mapping[id].message.as_ref().unwrap(),
                    generation: Some((pos+1, size)).filter(|_| add_generation),
                    is_diff_with_previous: false,
                });
                self.recurse(current, result, self.get_children(id));    
                current.pop();
            }
        }
    }

    fn get_all_conversations<'a>(&'a self) -> Vec<Vec<EnrichedMessage<'a>>> {
        let mut current: Vec<EnrichedMessage<'a>> = vec![];
        let mut result: Vec<Vec<EnrichedMessage<'a>>> = vec![];

        self.recurse(&mut current, &mut result, self.get_start_ids());

        result
    }
}

fn print_conversation(conv: Vec<EnrichedMessage<'_>>) {
    let skip_count = conv.iter().enumerate().map(|(i, x)| (i, x.is_diff_with_previous)).filter(|(_, x)| *x).map(|(i, _)| i).next().unwrap_or(0);
    if skip_count != 0 {
        println!("[SKIPPED {} MESSAGES THAT ARE THE SAME IN THE PREVIOUS DIALOGUE]\n\n", skip_count);
    }
    for (message_pos, message) in conv.into_iter().enumerate().skip(skip_count) {
        print!("({}) | ", message_pos+1);
        if let Some((pos, size)) = message.generation {
            print!("{{{}/{}}} ", pos, size);
        }
        match message.inner.author.role {
            AuthorRole::User => print!("[[USER]]: "),
            AuthorRole::Assistant => {
                match message.inner.metadata.model_slug {
                    Some(ModelSlug::Gpt4) => print!("[[GPT-4]]: "),
                    Some(ModelSlug::Davinci2) | None => print!("[[ChatGPT]]: "),
                }
            },
            AuthorRole::System => print!("[[SYSTEM]]: "),
        }
        println!("::::::::::::::::::::::::::::::::::::::::::");
        assert!(message.inner.content.parts.len() == 1);
        let part = &message.inner.content.parts[0];
        print!("{}", part);

        // fix interrupted code writing
        if part.split("```").count() % 2 == 0 {
            println!();
            println!("```");
        }

        if let Some(finish_details) = &message.inner.metadata.finish_details {
            if finish_details.kind == FinishDetailsType::Interrupted {
                println!("[INTERRUPTED]");
            }
            if finish_details.kind == FinishDetailsType::MaxTokens {
                println!("[MAX TOKENS REACHED]");
            }
        }

        println!();
        println!();
        println!();
    }
}

fn main() {
    let conversations1: Vec<Conversation> = serde_json::from_str(&std::fs::read_to_string("C:\\Users\\1\\Desktop\\chat2\\conversations.json").unwrap()).unwrap();   
    let conversations2: Vec<Conversation> = serde_json::from_str(&std::fs::read_to_string("C:\\Users\\1\\Desktop\\chat\\conversations.json").unwrap()).unwrap();   
    let conversations = conversations1.into_iter().rev().chain(conversations2.into_iter().rev()).collect::<Vec<_>>();

    let mut i = 0;
    for conv in conversations.into_iter() {
        println!("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-");
        println!("# {}", conv.title);
        println!("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-");

        let all_conversations = conv.get_all_conversations();
        for conv in all_conversations {
            i += 1;
            println!("------------------------------------------------------------");
            println!("## {}", i);
            println!("------------------------------------------------------------");
            print_conversation(conv);
        }
        println!();
        println!();
        println!();
    }
}
